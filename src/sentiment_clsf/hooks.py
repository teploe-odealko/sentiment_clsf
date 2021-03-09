from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal

from sentiment_clsf.pipelines import data_engineering as de
from sentiment_clsf.pipelines import data_science as ds
from sentiment_clsf.pipelines import api_prediction


class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        data_engineering_pipeline = de.create_pipeline()
        data_science_pipeline = ds.create_pipeline()
        api_prediction_pipeline = api_prediction.create_pipeline()
        return {
            "__default__": data_engineering_pipeline + data_science_pipeline,
            "de": data_engineering_pipeline,
            "ds": data_science_pipeline,
            "api_predict": api_prediction_pipeline,
        }

    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
