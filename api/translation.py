from deep_translator import GoogleTranslator

def sinhalaToEnglish(query: str) -> str:
    return GoogleTranslator(source="si", target="en").translate(query)

def englishToSinhala(text: str) -> str:
    return GoogleTranslator(source="en", target="si").translate(text)
