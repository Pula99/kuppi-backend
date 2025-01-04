from vertexai.generative_models import GenerativeModel
import logging
from services.translation_service import translate_text_google

logger = logging.getLogger(__name__)

def give_answer_sinhala(query, search_data):
    try:
        search_data_str = "\n".join([
            f"- Document: {item[0].metadata.get('source', 'Unknown Source')}, Relevance: {item[1]:.2f}\n{item[0].page_content}"
            for item in search_data
        ])

        prompt = (
            f"Using the following information:\n"
            f"{search_data_str}\n"
            f"### Provide an answer to the query: '{query}' ###\n"
            f"- Answer the query concisely and accurately based on the given information.\n"
            f"- Format the response in a simple and clear manner.\n"
            f"- Ensure the answer is the same and correct if translated from English to Sinhala.\n"
            f"- Present the answer in a point-wise format or in a way that is easy to understand and read, as this is for study purposes."
        )

        # logger.info("Prompt: %s", prompt)

        model = GenerativeModel("gemini-1.5-flash-001")
        responses = model.generate_content([prompt])

        if responses and responses.candidates:
            generated_answer = responses.candidates[0].content.parts[0].text
            logger.info("Generated Answer: %s", generated_answer)
        else:
            logger.warning("No valid candidates returned from the model.")
            return "No response generated."

        logger.info("\nGenerated Answer: %s\n", generated_answer)

        translated_answer = translate_text_google(generated_answer, target_language="si")
        if not translated_answer:
            logger.error("Translation failed for the generated answer.")
            return "An error occurred during translation."

        logger.info("Translated Answer: %s", translated_answer)

        corrected_answer = check_grammar(translated_answer)
        if not corrected_answer:
            logger.error("Grammar correction failed for the translated answer.")
            return "An error occurred during grammar correction."

        logger.info("Corrected Answer: %s", corrected_answer)
        return corrected_answer

    except Exception as e:
        logger.error(f"Error in give_answer: {e}")
        return "An error occurred while generating the answer."
    

def give_answer_english(query, search_data):
    logger.info("query: %s", query)
    try:
        search_data_str = "\n".join([
            f"- Document: {item[0].metadata.get('source', 'Unknown Source')}, Relevance: {item[1]:.2f}\n{item[0].page_content}"
            for item in search_data
        ])

        prompt = (
            f"Using the following information:\n"
            f"{search_data_str}\n"
            f"### Provide an answer to the query: '{query}' ###\n"
            f"- Answer the query concisely and accurately based on the given information.\n"
            f"- Format the response in a simple and clear manner.\n"
            f"- Present the answer in a point-wise format or in a way that is easy to understand and read, as this is for study purposes."
        )

        # logger.info("Prompt: %s", prompt)

        model = GenerativeModel("gemini-1.5-flash-001")
        responses = model.generate_content([prompt])

        if responses and responses.candidates:
            generated_answer = responses.candidates[0].content.parts[0].text
            logger.info("Generated Answer: %s", generated_answer)
        else:
            logger.warning("No valid candidates returned from the model.")
            return "No response generated."

        logger.info("\nGenerated Answer: %s\n", generated_answer)
        
        return generated_answer

    except Exception as e:
        logger.error(f"Error in give_answer: {e}")
        return "An error occurred while generating the answer."
    

def check_grammar(text):
    try:
        prompt = (
            f"Please correct the grammar of the following text in Sinhala:\n"
            f"{text}\n"
            f"### Ensure that the grammar follows the following rules:\n"
            f"1. Sentence Structure (Word Order): Sinhala follows a Subject-Object-Verb (SOV) structure.\n"
            f"2. Nouns: Nouns have gender and number (singular and plural).\n"
            f"3. Pronouns: Personal pronouns have different forms based on politeness and respect.\n"
            f"4. Verbs: Verbs change based on tense and formality.\n"
            f"5. Adjectives: Adjectives come before the noun they modify.\n"
            f"6. Particles: Particles are used to show emphasis or ask questions.\n"
            f"7. Postpositions: Postpositions show relationships like location and direction.\n"
            f"8. Honorifics: Honorifics are used for respect in formal settings.\n"
            f"9. Conjunctions: Conjunctions link ideas in a sentence.\n"
            f"10. Negation: Use 'නැහැ' or 'නො' to negate verbs.\n"
            f"11. Questions: Add 'ද?' at the end of a sentence to form a question.\n"
            f"12. Direct and Indirect Speech: Use direct speech for quoting and indirect speech for paraphrasing.\n"
            f"13. Relative Clauses: Use relative pronouns like 'ඉතා', 'මෙම', 'ඔබේ' to link clauses.\n"
            f"14. Honorifics in Verbs: Use honorifics to show respect in formal settings.\n"
            f"15. Use of 'අවශ්‍ය' (Necessary) and 'නැති' (Not Needed) to express necessity or lack thereof.\n"
            f"16. Comparatives and Superlatives: Use 'වැඩි' (more) or 'අඩු' (less) for comparisons.\n"
            f"17. Linking Verbs: Sinhala often omits the linking verb 'is/are'.\n"
            f"18. Future Conditional Sentences: Use 'ඔබ එන්නනම්' for expressing conditional actions.\n"
            f"19. Adverbial Conjunctions: Use conjunctions like 'එවක්' (then), 'නමුත්' (but) to connect ideas.\n"
            f"20. Possessive Pronouns: 'මගේ', 'ඔබේ', 'ඔවුන්ගේ' to express ownership.\n"
            f"21. Tense Agreement: Ensure the verb agrees with the tense.\n"
            f"22. Reduplication: Repeating words can be used for emphasis or habitual action.\n"
            f"23. Use of 'පහසු' (Easy) and 'අවශ්‍ය' (Hard) for difficulty level descriptions.\n"
            f"24. Use of 'සමඟ' (With) to show association or companionship.\n"
            f"25. Reflexive Verbs: Reflexive verbs end with 'වීම' (acting on oneself).\n"
            f"### dont menetion that these are the rules, just give the text to correct and response is generated by following these sinhala grammer rules ###"
        )

        # Generate a response using the generative model
        model = GenerativeModel("gemini-1.5-flash-001")
        responses = model.generate_content([prompt])

        if responses and responses.candidates:
            corrected_text = responses.candidates[0].content.parts[0].text
            return corrected_text
        else:
            return None

    except Exception as e:
        logger.error(f"Error in check_grammar: {e}")
        return None
