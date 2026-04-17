library(shiny)
library(xml2)
library(dplyr)
library(purrr)
library(stringr)
library(htmltools)
library(glue)
library(ragnar)
library(ellmer)
library(httr2)
library(ggplot2)
source("model.R")


taxi <- readRDS("taxi_sample.rds")
model_obj <- readRDS("tip_model.rds")

# =========================================================
# CONFIG
# =========================================================
HTML_PATH <- "RAG-File.html"
DB_PATH   <- "my_hf_rag.duckdb"
EMBEDDING_DIM <- 384

hf_token <- Sys.getenv("HUGGINGFACE_API_KEY")
if (hf_token == "") stop("HUGGINGFACE_API_KEY is not set.")

# optional: rebuild the vector store each launch
DB_PATH <- file.path(tempdir(), "my_hf_rag.duckdb")

if (file.exists(DB_PATH)) {
  unlink(DB_PATH, force = TRUE)
}
# =========================================================
# HTML CHUNK EXTRACTION
# =========================================================
build_html_chunks <- function(html_path) {
  doc <- read_html(html_path)
  divs <- xml_find_all(doc, "//div[@data-plot-id]")
  
  tibble(
    id     = xml_attr(divs, "data-plot-id"),
    tags   = xml_attr(divs, "data-tags"),
    text   = map_chr(divs, ~ str_squish(xml_text(.x))),
    html   = map_chr(divs, ~ as.character(.x))
  ) %>%
    mutate(
      tags = if_else(is.na(tags), "", tags),
      origin = paste0("html_chunk:", id),
      text = if_else(tags == "", text, paste(text, "\n\nTags: ", tags))
    ) %>%
    filter(!is.na(text), text != "")
}

html_chunks <- build_html_chunks(HTML_PATH)

# =========================================================
# HUGGING FACE EMBEDDINGS
# =========================================================
hf_embed <- function(texts) {
  embedding_dim <- 384
  
  hf_token <- Sys.getenv("HUGGINGFACE_API_KEY")
  if (hf_token == "") stop("HUGGINGFACE_API_KEY is not set.")
  
  if (length(texts) == 0) {
    return(matrix(numeric(0), nrow = 0, ncol = embedding_dim))
  }
  
  embed_one <- function(txt) {
    resp <- httr2::request(
      "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    ) |>
      httr2::req_headers(
        Authorization = paste("Bearer", hf_token),
        `Content-Type` = "application/json"
      ) |>
      httr2::req_body_json(list(inputs = txt)) |>
      httr2::req_timeout(60) |>
      httr2::req_perform()
    
    out <- httr2::resp_body_json(resp, simplifyVector = TRUE)
    
    if (is.numeric(out)) {
      vec <- out
    } else if (is.matrix(out)) {
      vec <- colMeans(out)
    } else if (is.array(out) && length(dim(out)) == 3) {
      token_mat <- out[1, , ]
      vec <- colMeans(token_mat)
    } else if (is.list(out)) {
      token_mat <- do.call(rbind, out)
      vec <- colMeans(token_mat)
    } else {
      stop("Unexpected embedding output shape from Hugging Face.")
    }
    
    if (length(vec) != embedding_dim) {
      stop(
        paste0(
          "Embedding length was ", length(vec),
          " but expected ", embedding_dim, "."
        )
      )
    }
    
    as.numeric(vec)
  }
  
  emb <- t(vapply(texts, embed_one, numeric(embedding_dim)))
  unname(emb)
}

# =========================================================
# VECTOR STORE
# =========================================================
store <- ragnar_store_create(
  DB_PATH,
  embed = hf_embed,
  version = 1,
  overwrite = TRUE
)

# ragnar expects chunk-like records with at least text.
# We keep origin for source display.
ragnar_store_insert(
  store,
  html_chunks %>% select(text, origin)
)

ragnar_store_build_index(store)

# =========================================================
# CHAT MODEL
# =========================================================
chat <- chat_huggingface(
  model = "meta-llama/Llama-3.1-8B-Instruct",
  system_prompt = paste(
    "You are a grounded assistant.",
    "Answer only from the retrieved context.",
    "If the answer is not in the context, say so clearly.",
    "When possible, mention the source."
  )
)

# =========================================================
# RETRIEVAL + RAG
# =========================================================
retrieve_chunks <- function(store, question, top_k = 5) {
  ragnar_retrieve_vss(
    store,
    query = question,
    top_k = top_k
  )
}

ask_rag <- function(question, store, chat, top_k = 5) {
  retrieved <- retrieve_chunks(store, question, top_k = top_k)
  
  if (nrow(retrieved) == 0) {
    return(list(
      answer = "I could not retrieve any relevant context.",
      retrieved = retrieved
    ))
  }
  
  origin_col <- if ("origin" %in% names(retrieved)) "origin" else NULL
  
  context_parts <- vapply(
    seq_len(nrow(retrieved)),
    function(i) {
      txt <- retrieved$text[i]
      src <- if (!is.null(origin_col)) retrieved[[origin_col]][i] else "unknown"
      paste0("SOURCE: ", src, "\n", txt)
    },
    character(1)
  )
  
  context <- paste(context_parts, collapse = "\n\n--------------------\n\n")
  
  prompt <- paste0(
    "Answer the question using only the context below.\n\n",
    "Question:\n", question, "\n\n",
    "Context:\n", context, "\n\n",
    "Instructions:\n",
    "- Be concise.\n",
    "- If the answer is not in the context, say that clearly.\n",
    "- Mention the source when helpful.\n"
  )
  
  answer <- chat$chat(prompt)
  
  list(
    answer = as.character(answer),
    retrieved = retrieved
  )
}

# helper to map retrieved origin back to original html snippet
lookup_html_by_origin <- function(origin_value, html_chunks_tbl) {
  idx <- match(origin_value, html_chunks_tbl$origin)
  if (is.na(idx)) return(NULL)
  html_chunks_tbl$html[idx]
}

# =========================================================
# UI
# =========================================================
ui <- navbarPage(
  "Yellow Taxi App",
  
  tabPanel(
    "RAG Q&A",
    fluidPage(
      tags$head(
        tags$script(HTML("
          Shiny.addCustomMessageHandler('start_voice', function(message) {
            try {
              var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
              if (!SpeechRecognition) {
                alert('Speech recognition not supported in this browser.');
                return;
              }
              var recognition = new SpeechRecognition();
              recognition.lang = 'en-US';
              recognition.interimResults = false;
              recognition.maxAlternatives = 1;
              recognition.onresult = function(event) {
                var transcript = event.results[0][0].transcript;
                Shiny.setInputValue('voice_text', transcript, {priority: 'event'});
              };
              recognition.start();
            } catch (e) {
              console.log(e);
              alert('Error starting speech recognition.');
            }
          });
        "))
      ),
      
      sidebarLayout(
        sidebarPanel(
          textInput("query", "Ask a question:", ""),
          numericInput("top_k", "Number of chunks to retrieve", value = 5, min = 1, max = 10),
          actionButton("ask", "Answer"),
          br(), br(),
          actionButton("voice", "🎤 Speak question"),
          helpText("Voice input fills the text box if your browser supports speech recognition.")
        ),
        mainPanel(
          h4("RAG Answer (Hugging Face + Ragnar)"),
          verbatimTextOutput("answer_text"),
          tags$hr(),
          h4("Top Retrieved Source"),
          uiOutput("answer_html"),
          tags$hr(),
          h4("Retrieved Sources"),
          tableOutput("retrieved_table")
        )
      )
    )
  ),
  
  tabPanel(
    "Tip Prediction",
    fluidPage(
      sidebarLayout(
        sidebarPanel(
          numericInput("trip_duration_mins", "Trip Duration (mins)", value = 15, min = 1),
          numericInput("fare_amount", "Fare Amount ($)", value = 15, min = 0.01),
          selectInput(
            "Ratecode_name",
            "Rate Code",
            choices = model_obj$levels$Ratecode_name,
            selected = model_obj$levels$Ratecode_name[1]
          ),
          actionButton("predict_btn", "Predict Tip")
        ),
        mainPanel(
          h4("Predicted Tip"),
          verbatimTextOutput("predicted_tip")
        )
      )
    )
  ),
  tabPanel(
    "Plots",
    fluidPage(
      sidebarLayout(
        sidebarPanel(
          selectInput(
            "plot_pickup_time",
            "Pickup Time of Day",
            choices = c("All", sort(unique(as.character(taxi$pickup_time_of_day)))),
            selected = "All"
          ),
          selectInput(
            "plot_pickup_location",
            "Pickup Borough",
            choices = c("All", sort(unique(as.character(taxi$Borough_pickup)))),
            selected = "All"
          ),
          selectInput(
            "plot_dropoff_location",
            "Dropoff Borough",
            choices = c("All", sort(unique(as.character(taxi$Borough_dropoff)))),
            selected = "All"
          ),
          selectInput(
            "plot_ratecode",
            "Rate Code",
            choices = c("All", sort(unique(as.character(taxi$Ratecode_name)))),
            selected = "All"
          )
        ),
        mainPanel(
          h4("Average Tip by Pickup Time of Day"),
          plotOutput("plot_tip_by_time", height = "350px"),
          
          tags$hr(),
          
          h4("Average Tip by Pickup Borough"),
          plotOutput("plot_tip_by_pickup", height = "350px"),
          
          tags$hr(),
          
          h4("Average Tip by Dropoff Borough"),
          plotOutput("plot_tip_by_dropoff", height = "350px")
        )
      )
    )
  )
)

# =========================================================
# SERVER
# =========================================================
server <- function(input, output, session) {
  
  observeEvent(input$voice_text, {
    updateTextInput(session, "query", value = input$voice_text)
  })
  
  observeEvent(input$voice, {
    session$sendCustomMessage("start_voice", list())
  })
  
  rag_result <- eventReactive(input$ask, {
    req(input$query)
    ask_rag(
      question = input$query,
      store = store,
      chat = chat,
      top_k = input$top_k
    )
  })
  
  output$answer_text <- renderText({
    res <- rag_result()
    req(res)
    res$answer
  })
  
  output$answer_html <- renderUI({
    res <- rag_result()
    req(res)
    
    retrieved <- res$retrieved
    if (nrow(retrieved) == 0) {
      return(HTML("<em>No retrieved content to display.</em>"))
    }
    
    html_blocks <- lapply(seq_len(nrow(retrieved)), function(i) {
      origin_val <- if ("origin" %in% names(retrieved)) retrieved$origin[i] else NULL
      if (is.null(origin_val)) return(NULL)
      
      html_snippet <- lookup_html_by_origin(origin_val, html_chunks)
      if (is.null(html_snippet) || is.na(html_snippet) || html_snippet == "") return(NULL)
      
      tagList(
        tags$div(
          style = "margin-bottom: 24px;",
          tags$h5(paste("Relevant chart", i)),
          HTML(html_snippet)
        )
      )
    })
    
    do.call(tagList, html_blocks)
  })
  
  output$retrieved_table <- renderTable({
    res <- rag_result()
    req(res)
    
    retrieved <- res$retrieved
    if (nrow(retrieved) == 0) return(NULL)
    
    out <- retrieved
    
    keep_cols <- intersect(c("origin", "text", "distance", "score"), names(out))
    out <- out[, keep_cols, drop = FALSE]
    
    if ("text" %in% names(out)) {
      out$text <- substr(out$text, 1, 160)
    }
    
    out
  }, striped = TRUE, bordered = TRUE)
  
  tip_prediction <- eventReactive(input$predict_btn, {
    fare_val <- suppressWarnings(as.numeric(input$fare_amount))
    
    new_data <- data.frame(
      trip_duration_mins = input$trip_duration_mins,
      Ratecode_name = input$Ratecode_name,
      stringsAsFactors = FALSE
    )
    
    pred_pct <- predict_tip_percent(model_obj, new_data)
    
    list(
      tip_percent = pred_pct,
      tip_amount = if (!is.na(fare_val) && nzchar(input$fare_amount)) pred_pct * fare_val else NA_real_
    )
  })
  
  output$predicted_tip <- renderText({
    req(tip_prediction())
    
    res <- tip_prediction()
    pct_text <- paste0("Predicted tip percent: ", round(100 * res$tip_percent, 1), "%")
    
    if (is.na(res$tip_amount)) {
      pct_text
    } else {
      paste0(
        pct_text,
        "\nPredicted tip amount: $",
        round(res$tip_amount, 2)
      )
    }
  })
  
  
  
  
  
  filtered_plot_data <- reactive({
    df <- taxi
    
    if (input$plot_pickup_time != "All") {
      df <- df %>% filter(as.character(pickup_time_of_day) == input$plot_pickup_time)
    }
    
    if (input$plot_pickup_location != "All") {
      df <- df %>% filter(as.character(Borough_pickup) == input$plot_pickup_location)
    }
    
    if (input$plot_dropoff_location != "All") {
      df <- df %>% filter(as.character(Borough_dropoff) == input$plot_dropoff_location)
    }
    
    if (input$plot_ratecode != "All") {
      df <- df %>% filter(as.character(Ratecode_name) == input$plot_ratecode)
    }
    
    df
  })
  
  output$plot_tip_by_time <- renderPlot({
    df <- filtered_plot_data()
    req(nrow(df) > 0)
    
    plot_df <- df %>%
      group_by(pickup_time_of_day) %>%
      summarise(
        avg_tip = mean(tip_amount, na.rm = TRUE),
        .groups = "drop"
      )
    
    ggplot(plot_df, aes(x = pickup_time_of_day, y = avg_tip)) +
      geom_col() +
      labs(
        x = "Pickup Time of Day",
        y = "Average Tip Amount",
        title = "Average Tip by Pickup Time of Day"
      ) +
      theme_minimal()
  })
  
  output$plot_tip_by_pickup <- renderPlot({
    df <- filtered_plot_data()
    req(nrow(df) > 0)
    
    plot_df <- df %>%
      group_by(Borough_pickup) %>%
      summarise(
        avg_tip = mean(tip_amount, na.rm = TRUE),
        .groups = "drop"
      )
    
    ggplot(plot_df, aes(x = Borough_pickup, y = avg_tip)) +
      geom_col() +
      labs(
        x = "Pickup Borough",
        y = "Average Tip Amount",
        title = "Average Tip by Pickup Borough"
      ) +
      theme_minimal()
  })
  
  output$plot_tip_by_dropoff <- renderPlot({
    df <- filtered_plot_data()
    req(nrow(df) > 0)
    
    plot_df <- df %>%
      group_by(Borough_dropoff) %>%
      summarise(
        avg_tip = mean(tip_amount, na.rm = TRUE),
        .groups = "drop"
      )
    
    ggplot(plot_df, aes(x = Borough_dropoff, y = avg_tip)) +
      geom_col() +
      labs(
        x = "Dropoff Borough",
        y = "Average Tip Amount",
        title = "Average Tip by Dropoff Borough"
      ) +
      theme_minimal()
  })
}

shinyApp(ui, server)