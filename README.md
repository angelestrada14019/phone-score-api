# Analizador de Teléfonos

Este proyecto es una aplicación web construida con FastAPI que evalúa smartphones basándose en sus especificaciones técnicas y proporciona métricas de rendimiento, rango de precio estimado y recomendaciones.

## APIs Utilizadas

* **API de Evaluación de Smartphones:** Esta es la API principal del proyecto, construida con FastAPI. Utiliza un modelo de machine learning (`phone_analyzer_pro.pkl`) o un método de cálculo alternativo para evaluar las especificaciones de los teléfonos.
### Endpoints de la API de Evaluación de Smartphones

* **GET `/`**
    * **Descripción:** Este endpoint devuelve un mensaje de estado (`{"status": "ok", "message": "Smartphone Evaluation API is running"}`) para indicar que la API está funcionando correctamente.
    * **Respuesta:**


* **GET `/smartphones/samples`**
    * **Descripción:** Este endpoint devuelve una lista de ejemplos de smartphones en el formato `SmartphoneInput`.
    * **Respuesta:** Una lista de objetos con el siguiente formato:

*   **POST `/smartphones/evaluate`**
    *   **Descripción:** Este es el endpoint principal para evaluar un smartphone. Recibe las especificaciones de un smartphone y devuelve una evaluación completa.
    *   **Parámetros de Entrada (en el cuerpo de la solicitud JSON):**
        *   `internal_storage` (int): Almacenamiento interno en GB.
        *   `storage_ram` (int): Memoria RAM en GB.
        *   `expandable_storage` (Union[float, str]): Almacenamiento expandible en TB o "NA".
        *   `primary_camera` (str): Configuración de la cámara principal (ej. "108MP + 12MP + 5MP + 5MP").
        *   `display` (str): Tipo y resolución de la pantalla (ej. "Full HD+ Dynamic AMOLED 2X").
        *   `network` (str): Tecnologías de red soportadas (ej. "5G, 4G, 3G, 2G").
        *   `battery` (int): Capacidad de la batería en mAh.
    *   **Respuesta:** Un objeto que representa la evaluación completa del smartphone, incluyendo `id`, `overall_score`, `performance_category`, `price_range`, `user_recommendation`, y un objeto `metrics` anidado con `gaming_potential`, `battery_performance`, `photography`, y `display_quality`.


    * **Parámetros de Entrada (en el cuerpo de la solicitud JSON):**


## Herramientas Empleadas

* **Python:** Lenguaje de programación principal para el desarrollo de la aplicación.
* **FastAPI:** Framework web para construir la API de la aplicación.
* **Flask:** Microframework web para construir la API de la aplicación.
* **Railway:** Plataforma de despliegue en la nube.
* **Git:** Sistema de control de versiones.

## Despliegue en Railway

Para desplegar este proyecto en Railway, sigue los siguientes pasos:

1. **Crea una cuenta en Railway:** Si aún no tienes una, regístrate en [https://railway.app/](https://railway.app/).
2. **Conecta tu repositorio de Git:** En el panel de Railway, crea un nuevo proyecto y conecta tu repositorio de Git donde se encuentra este código.
3. **Configura la aplicación:**
    * Railway debería detectar automáticamente que es una aplicación Python.
    * Asegúrate de que el comando de inicio esté configurado para ejecutar la aplicación Flask. Por ejemplo: `gunicorn main:app`.
    * Configura las variables de entorno necesarias (si las hay).
4. **Despliega:** Railway construirá la imagen de tu aplicación y la desplegará automáticamente.
5. **Verifica el despliegue:** Una vez que el despliegue sea exitoso, Railway te proporcionará una URL donde podrás acceder a tu aplicación.

**Nota:** Asegúrate de que tu archivo `requirements.txt` liste todas las dependencias necesarias para que Railway pueda instalarlas correctamente durante el proceso de construcción.
