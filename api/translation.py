from deep_translator import GoogleTranslator

def sinhalaToEnglish(query: str) -> str:
    translator = GoogleTranslator(source="si", target="en")
    return translator.translate(query)

def englishToSinhala(text: str) -> str:
    translator = GoogleTranslator(source="en", target="si")
    return translator.translate(text)
