import hls4ml
import yaml

with open('DC.yml', 'r') as fin:
    cfg = yaml.load(fin)

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.config.writer.write_hls(hls_model)