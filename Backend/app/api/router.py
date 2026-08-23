import gradio as gr

from app.damage_pipeline import analyze


def create_interface():
    return gr.Interface(
        fn=analyze,
        inputs=gr.Image(type="filepath", label="Upload car image"),
        outputs=[
            gr.Image(type="pil", label="segmented_image"),
            gr.Textbox(label="Classification"),
            gr.Textbox(label="Detected_damage"),
            gr.Markdown(label="Damage summary", show_label=True, container=True),
        ],
        title="car damage inspector",
    )