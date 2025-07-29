import reflex as rx


config = rx.Config(
    app_name="chat",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    # api_url="http://localhost:8000",
    # api_url="https://secure-pig-driving.ngrok-free.app:8000"
)
