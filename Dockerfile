FROM python:3.11.4
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
RUN set FLASK_APP=app.py
CMD [ "flask","run","--host=0.0.0.0","--port=5000" ]
