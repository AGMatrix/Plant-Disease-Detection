"""
Plant Disease Detection - Hugging Face Space
"""
import gradio as gr
from src.predict import DiseasePredictor
import os

def create_interface():
    """Create Gradio interface"""

    predictor = DiseasePredictor('best_model.pth', 'classes.json')

    def predict_disease(image):
        if image is None:
            return "Please upload an image!"

        temp_path = 'temp_upload.jpg'
        image.save(temp_path)

        results = predictor.predict(temp_path, top_k=5)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        output = "**Plant Disease Detection Results**\n\n"
        output += f"**Primary Diagnosis:** {results[0]['disease']}\n"
        output += f"**Confidence:** {results[0]['confidence']:.2f}%\n\n"

        if results[0]['confidence'] > 90:
            output += "**High Confidence**\n\n"
        elif results[0]['confidence'] > 70:
            output += "**Moderate Confidence**\n\n"
        else:
            output += "**Low Confidence - Multiple Possibilities**\n\n"

        if len(results) > 1:
            output += "**Other Possibilities:**\n\n"
            for i, result in enumerate(results[1:], 2):
                output += f"{i}. {result['disease']} - {result['confidence']:.2f}%\n"

        return output

    interface = gr.Interface(
        fn=predict_disease,
        inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
        outputs=gr.Markdown(label="Detection Results"),
        title="Plant Disease Detection System",
        description="""
        ### Plant Disease Identification

        Upload a clear image of a plant leaf to detect diseases.

        **Tips for best results:**
        - Use well-lit, focused images
        - Capture the affected area clearly
        - Avoid blurry or very dark images
        """,
        theme="soft"
    )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
