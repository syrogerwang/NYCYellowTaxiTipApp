library(shiny)
library(xml2)
library(dplyr)
library(purrr)
library(stringr)
library(stringdist)
library(htmltools)
library(glue)
source("model.R")

model_obj <- readRDS("tip_model.rds")

# ==== CONFIG ====
HTML_PATH <- "RAG-File.html"

# ==== RAG INDEX BUILDING ====
build_rag_index <- function(html_path) {
  doc <- read_html(html_path)
  
  divs <- xml_find_all(doc, "//div[@data-plot-id]")
  
  tibble(
    id   = xml_attr(divs, "data-plot-id"),
    tags = xml_attr(divs, "data-tags"),
    type = "chunk",
    text = map_chr(divs, ~ str_squish(xml_text(.x))),
    html = map_chr(divs, ~ as.character(.x))
  ) %>%
    mutate(
      tags = if_else(is.na(tags), "", tags),
      search_text = tolower(paste(text, tags, id, sep = " "))
    )
}

retrieve_top_chunk <- function(index, query, k = 1) {
  if (is.null(query) || query == "") return(NULL)
  
  q <- tolower(query)
  texts <- index$search_text
  
  d <- stringdistmatrix(q, texts, method = "cosine")
  scores <- as.numeric(d[1, ])
  best <- order(scores)[seq_len(min(k, nrow(index)))]
  index[best, , drop = FALSE]
}

rag_index <- build_rag_index(HTML_PATH)

# ==== LOCAL ANSWER FROM CONTEXT (NO LLM) ====
answer_from_context <- function(question, context_text) {
  if (is.null(context_text) || context_text == "") {
    return("No relevant content found in the HTML index.")
  }
  
  sentences <- unlist(strsplit(context_text, "(?<=[.!?])\\s+", perl = TRUE))
  sentences <- str_squish(sentences)
  sentences <- sentences[nchar(sentences) > 0]
  
  if (length(sentences) == 0) return(context_text)
  
  q_words <- str_to_lower(str_split(question, "\\W+", simplify = TRUE))
  q_words <- q_words[q_words != ""]
  
  score_sentence <- function(s) {
    s_words <- str_to_lower(str_split(s, "\\W+", simplify = TRUE))
    s_words <- s_words[s_words != ""]
    length(intersect(q_words, s_words))
  }
  
  scores <- vapply(sentences, score_sentence, numeric(1))
  
  if (all(scores == 0)) {
    best_sentences <- head(sentences, 3)
  } else {
    ord <- order(scores, decreasing = TRUE)
    best_sentences <- sentences[head(ord, 3)]
  }
  
  paste(best_sentences, collapse = " ")
}

# ==== SHINY APP ====

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
          actionButton("ask", "Answer"),
          br(), br(),
          actionButton("voice", "🎤 Speak question"),
          helpText("Voice input fills the text box if your browser supports speech recognition.")
        ),
        mainPanel(
          h4("RAG Interpretation (Local Heuristic — No LLM)"),
          verbatimTextOutput("answer_text"),
          tags$hr(),
          h4("Relevant HTML Chunk"),
          uiOutput("answer_html")
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
  )
)

server <- function(input, output, session) {
  
  observeEvent(input$voice_text, {
    updateTextInput(session, "query", value = input$voice_text)
  })
  
  observeEvent(input$voice, {
    session$sendCustomMessage("start_voice", list())
  })
  
  rag_result <- eventReactive(input$ask, {
    req(input$query)
    retrieve_top_chunk(rag_index, input$query, k = 1)
  })
  
  output$answer_text <- renderText({
    res <- rag_result()
    if (is.null(res) || nrow(res) == 0) {
      return("No relevant content found in the HTML index.")
    }
    
    context_text <- res$text[1]
    answer_from_context(input$query, context_text)
  })
  
  output$answer_html <- renderUI({
    res <- rag_result()
    if (is.null(res) || nrow(res) == 0) {
      return(HTML("<em>No content to display.</em>"))
    }
    HTML(res$html[1])
  })
  
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
}

shinyApp(ui, server)
