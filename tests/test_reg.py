
if __name__ == "__main__":
    # read config file from configs/model/dlib_resnet.yaml
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" )
    output_path = path / "outputs"
    print("paths", path, config_path, output_path)
    # datamodule = hydra.utils.instantiate(cfg.data)
    # datamodule.prepare_data()
    # datamodule.setup()
    # loader = datamodule.train_dataloader()
    # # # loader
    # batch = next(iter(loader))
    def test_net(cfg):
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
    # # loader
        batch = next(iter(loader))
        net = hydra.utils.instantiate(cfg.model.net)
        print("*"*20+" net "+"*"*20, "\n", net)
        output = net(batch)
        print("output", output.shape)
    def test_module(cfg):
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
    # # loader
        batch = next(iter(loader))
        module = hydra.utils.instantiate(cfg.model)
        output = module(batch)
        print("module output", output.shape)
    
    @hydra.main(version_base="1.3", config_path=config_path, config_name="test.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        # test_net(cfg)
        test_module(cfg)

    main()