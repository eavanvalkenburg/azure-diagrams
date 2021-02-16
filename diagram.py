from diagrams import Diagram, Cluster, Edge, Node
from diagrams.azure.compute import ContainerInstances, ContainerRegistries, FunctionApps
from diagrams.azure.web import AppServiceEnvironments
from diagrams.azure.storage import DataLakeStorage
from diagrams.azure.database import CacheForRedis, SQLDatabases
from diagrams.azure.ml import MachineLearningServiceWorkspaces

graph_attr = {
    "layout": "neato",
    "mode": "major",
    "overlap": "voronoi",
    "sep": "0.0",
    "splines": "spline",
    "dpi": "192",
}


with Diagram(
    "ML integrated into website",
    show=False,
    graph_attr=graph_attr,
):
    storage = DataLakeStorage("Historical Data", pin="true", pos="0,4")

    with Cluster("Modelling Pipeline"):
        ml_ws = MachineLearningServiceWorkspaces("ML Workspace", pin="true", pos="0,8")
        acr = ContainerRegistries("Container Registry", pin="true", pos="3,8")

    with Cluster("Data pipeline"):
        cache = CacheForRedis("Real-time DB", pin="true", pos="6,0")
        cache_function = FunctionApps(
            "Update Cache Function", width="0.7", pin="true", pos="0,0"
        )

    with Cluster("Prediction pipeline"):
        website = AppServiceEnvironments("Website", pin="true", pos="9,4")
        scoring_container = ContainerInstances(
            "Real-time Scoring API", pin="true", pos="3,4"
        )
        predict_function = FunctionApps(
            "User profile enrichment", width="0.7", pin="true", pos="6,4"
        )

    # storage connections
    storage << Edge(label="Read Training data", color="blue") << ml_ws >> Edge(
        label="Image with trained model", style="dashed"
    ) >> acr
    storage << Edge(label="Read Daily", color="blue") << cache_function >> Edge(
        label="Daily", color="blue"
    ) >> cache

    # container image
    acr >> Edge(style="dashed", label="Deploy image") >> scoring_container

    # prediction
    predict_function >> Edge(forward=True, reverse=True, color="green") >> cache
    scoring_container >> Edge(
        label="Prediction", forward=True, reverse=True, color="green"
    ) >> predict_function >> Edge(
        label="Prediction", forward=True, reverse=True, color="green"
    ) >> website
