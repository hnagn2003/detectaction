import copy
from collections import OrderedDict
from mmengine.evaluator import BaseMetric
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import METRICS
import torch

@METRICS.register_module()
class AccuracyMetric(BaseMetric):
    def __init__(self, topk=(1, 5), collect_device='cpu', prefix='acc'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topk = topk

    def process(self, data_batch, data_samples):
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            scores = data_sample['pred_scores']['item'].cpu().numpy()
            label = data_sample['gt_labels']['item'].item()
            result['scores'] = scores
            result['label'] = label
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        eval_results = OrderedDict()
        labels = [res['label'] for res in results]
        scores = [res['scores'] for res in results]
        topk_acc = top_k_accuracy(scores, labels, self.topk)
        for k, acc in zip(self.topk, topk_acc):
            eval_results[f'topk{k}'] = acc
        return eval_results


if __name__ == "__main__":
    # read config file from configs/model/dlib_resnet.yaml
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    output_path = path / "outputs"
    print("paths", path, config_path, output_path)

    # def test_net(cfg):
    #     net = hydra.utils.instantiate(cfg.net)
    #     print("*"*20+" net "+"*"*20, "\n", net)
    #     output = net(torch.randn(1, 10, 3, 224, 224))
    #     print("output", output.shape)

    def test_module(cfg):
        metric_cfg = dict(type=cfg.metrics_type, topk=cfg.topk)
        metric = METRICS.build(metric_cfg)
        data_samples = [d.to_dict() for d in predictions]
        # module = hydra.utils.instantiate(cfg)
        # output = module(torch.randn(1, 10, 3, 224, 224))
        # print("module output", output.shape)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="recog.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        # test_net(cfg)
        test_module(cfg)

    main()
