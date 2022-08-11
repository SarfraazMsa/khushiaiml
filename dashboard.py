from explainerdashboard import ClassifierExplainer, ExplainerDashboard

explainer = ClassifierExplainer.from_file("explainer.joblib")
# you can override params during load from_config:
db = ExplainerDashboard.from_config(explainer, "dashboard.yaml",  logins=[['kb_ai', 'ai_for_impact'],['kb_research','ai_for_social_good']],db_users=dict(db1=['kb_ai'], db2=['kb_research']),title="Confirmed SAM Cases Model")

db.run(host='0.0.0.0', port=9050, use_waitress=True)






