"""Launch Basic Bot locally."""

from dotenv import load_dotenv
load_dotenv()

from web.server import create_local_app

app = create_local_app()
app.run(debug=True, port=5084)
