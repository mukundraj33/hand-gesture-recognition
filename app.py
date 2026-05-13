import gradio as gr


pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        from utils.pipeline import GestureRecognitionPipeline

        pipeline = GestureRecognitionPipeline()
    return pipeline


def recognize_gesture(image):
    annotated_image, prediction = get_pipeline().predict(image)
    return annotated_image, prediction


with gr.Blocks(title="Hand Gesture Recognition") as demo:
    gr.Markdown("# Hand Gesture Recognition")
    gr.Markdown("Use your browser webcam or upload an image to recognize hand gestures.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Webcam input",
            )
            recognize_button = gr.Button("Recognize Gesture", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Annotated output")
            prediction_text = gr.Textbox(label="Prediction", interactive=False)

    recognize_button.click(
        fn=recognize_gesture,
        inputs=input_image,
        outputs=[output_image, prediction_text],
    )
    input_image.change(
        fn=recognize_gesture,
        inputs=input_image,
        outputs=[output_image, prediction_text],
    )


if __name__ == "__main__":
    demo.launch()
