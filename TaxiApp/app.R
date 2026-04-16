# app.R
library(shiny)
library(xml2)
library(dplyr)
library(purrr)
library(stringr)
library(stringdist)
library(htmltools)
library(httr)
library(jsonlite)
library(glue)

# ==== CONFIG ====
HTML_PATH <- "report.html"   # your HTML file
HF_MODEL  <- "mistralai/Mistral-7B-Instruct-v0.2"  # change if you like

# ==== RAG INDEX BUILDING ====
build_rag_index <- function(html_path) {
  doc <- read_html(html_path)
  
  divs <- xml_find_all(doc, "//div[@id]")
  
  tibble(
    id   = xml_attr(divs, "data-plot-id"),
    tags = xml_attr(divs, "data-tags"),
    type = "chunk",
    text = map_chr(divs, ~ {
      # Use visible text inside the div as semantic content
      str_squish(xml_text(.x))
    }),
    html = map_chr(divs, ~ as.character(as.tags(.x)))
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

# ==== HUGGING FACE LLM CALL ====

call_hf_llm <- function(question, context, model = HF_MODEL) {
  token <- Sys.getenv("HF_API_TOKEN")
  if (token == "") {
    return("HF_API_TOKEN not set. Please set Sys.setenv(HF_API_TOKEN = '...').")
  }
  
  url <- glue("https://api-inference.huggingface.co/models/{model}")
  
  prompt <- glue(
    "You are an analyst answering questions based only on the provided context.\n\n",
    "Question:\n{question}\n\n",
    "Context (HTML-derived text):\n{context}\n\n",
    "Answer in 3–6 concise sentences, focusing on the key insight."
  )
  
  body <- list(
    inputs = prompt,
    parameters = list(
      max_new_tokens = 256,
      temperature = 0.2
    )
  )
  
  res <- POST(
    url,
    add_headers(Authorization = paste("Bearer", token)),
    body = toJSON(body, auto_unbox = TRUE),
    encode = "raw"
  )
  
  if (http_error(res)) {
    return(paste("Error from Hugging Face API:", status_code(res)))
  }
  
  txt <- content(res, as = "text", encoding = "UTF-8")
  parsed <- fromJSON(txt)
  
  # Handle both list-of-generations and plain text formats
  if (is.list(parsed) && !is.null(parsed[[1]]$generated_text)) {
    return(parsed[[1]]$generated_text)
  } else if (is.character(parsed)) {
    return(parsed)
  } else {
    return("Unexpected response format from Hugging Face API.")
  }
}

# ==== SHINY APP ====

ui <- fluidPage(
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
  
  titlePanel("RAG Q&A over HTML (with LLM)"),
  
  sidebarLayout(
    sidebarPanel(
      textInput("query", "Ask a question:", ""),
      actionButton("ask", "Answer"),
      br(), br(),
      actionButton("voice", "🎤 Speak question"),
      helpText("Voice input fills the text box if your browser supports speech recognition.")
    ),
    mainPanel(
      h4("LLM interpretation"),
      verbatimTextOutput("answer_text"),
      tags$hr(),
      h4("Relevant HTML chunk (table/image/text)"),
      uiOutput("answer_html")
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
  
  llm_answer <- reactive({
    res <- rag_result()
    if (is.null(res) || nrow(res) == 0) {
      return("No relevant content found in the HTML index.")
    }
    context_text <- res$text[1]
    call_hf_llm(input$query, context_text)
  })
  
  output$answer_text <- renderText({
    llm_answer()
  })
  
  output$answer_html <- renderUI({
    res <- rag_result()
    if (is.null(res) || nrow(res) == 0) {
      return(HTML("<em>No content to display.</em>"))
    }
    HTML(res$html[1])
  })
}

shinyApp(ui, server)
