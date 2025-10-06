# Importamos la clase Ollama desde la comunidad de LangChain
from langchain_community.llms import Ollama

def main():
    """
    Función principal para probar la conexión con el modelo phi3:mini de Ollama.
    """
    print("Iniciando prueba de conexión con Ollama (modelo: phi3:mini)...")

    # 1. Asegúrate de que Ollama esté corriendo en tu máquina.
    #    Puedes verificarlo abriendo una terminal y ejecutando: ollama list

    try:
        # 2. Creamos una instancia del LLM, especificando el modelo a usar.
        #    *** AQUÍ ESTÁ EL CAMBIO IMPORTANTE ***
        #    Usamos 'phi3:mini' que se ajusta a tu hardware.
        llm = Ollama(model="phi3:mini")

        # 3. Hacemos una pregunta simple para probar la invocación.
        print("Enviando pregunta al modelo...")
        pregunta = "¿Por qué el cielo es azul?"
        respuesta = llm.invoke(pregunta)

        # 4. Imprimimos la respuesta obtenida.
        print("\n" + "="*50)
        print(f"Pregunta: {pregunta}")
        print(f"Respuesta del modelo: {respuesta}")
        print("="*50)
        print("\n¡Conexión exitosa! LangChain se está comunicando con tu modelo 'phi3:mini'.")

    except Exception as e:
        print(f"\nError: No se pudo conectar con Ollama. Detalles del error:")
        print(e)
        print("\nPor favor, asegúrate de que el servicio de Ollama esté en ejecución.")
        print("Puedes iniciarlo desde la aplicación de escritorio o ejecutando 'ollama serve' en la terminal.")

if __name__ == "__main__":
    main()