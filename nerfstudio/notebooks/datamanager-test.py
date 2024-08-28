from nerfstudio.data.datamanagers.ad_datamanager import ADDataManagerConfig, ADDataManager

config = ADDataManagerConfig()
manager = ADDataManager(config, device="cpu")
