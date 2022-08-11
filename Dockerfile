FROM python:3.8

RUN pip install explainerdashboard numpy sqlalchemy numpy psycopg2 statsmodels openpyxl seaborn fancyimpute sklearn matplotlib imblearn
COPY generate_dashboard.py ./
COPY dashboard.py ./
COPY Master_children_dataset_Udaipur.csv ./
COPY malnutrition.xlsx ./
RUN python generate_dashboard.py

EXPOSE 8050
CMD ["python", "./dashboard.py"]


