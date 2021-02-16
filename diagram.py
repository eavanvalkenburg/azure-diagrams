from diagrams import Diagram, Cluster, Edge
from diagrams.azure.compute import ContainerInstances, ContainerRegistries, FunctionApps
from diagrams.azure.web import AppServiceEnvironments
from diagrams.azure.storage import DataLakeStorage
from diagrams.azure.database import CacheForRedis, SQLDatabases
from diagrams.azure.ml import MachineLearningServiceWorkspaces

with Diagram("ML integrated into website", show=False, curvestyle="ortho"):
    storage = DataLakeStorage("Historical Data")

    with Cluster("Modelling Pipeline"):
        ml_ws = MachineLearningServiceWorkspaces("ML Workspace")
        acr = ContainerRegistries("Container Registry")
        modelling = [ml_ws, acr]
        storage >> ml_ws >> Edge(label="Image with trained model") >> acr

    with Cluster("Data pipeline"):
        cache = CacheForRedis("Real-time DB")
        cache_function = FunctionApps("Update Cache Function")
        data = [cache_function, cache]
        storage >> Edge(label="Daily") >> cache_function >> Edge(label="Daily") >> cache

    with Cluster("Prediction pipeline"):
        website = AppServiceEnvironments("Website")
        scoring_container = ContainerInstances("Real-time Scoring API")
        predict_function = FunctionApps("User profile enrichment")
        prediction = [website, predict_function, scoring_container]
        scoring_container >> Edge(
            label="Prediction", forward=True, reverse=True
        ) >> predict_function >> Edge(
            label="Prediction", forward=True, reverse=True
        ) >> website

    acr >> Edge(style="dashed", label="Deploy image") >> scoring_container
    predict_function >> Edge(forward=True, reverse=True) >> cache
