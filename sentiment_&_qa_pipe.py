from transformers import pipeline

# Text generation
# generator = pipeline("text-generation")

# # Question answering  
# qa = pipeline("question-answering")

# # Summarisation
# summarizer = pipeline("summarization")

# # Translation
# translator = pipeline("translation_en_to_fr")

# classifier = pipeline("sentiment-analysis")
# result = classifier("I love building AI projects!")
# print(result)

# qa = pipeline("question-answering")

# result = qa(
#     question="What is the capital of France?",
#     context="France is a country in Europe. Its capital city is Paris, which is known for the Eiffel Tower."
# )

# print(result)

# Loading a SPECIFIC model by name from HuggingFace Hub
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

result = classifier("This RAG project was really challenging but I learned a lot!")
print(result)