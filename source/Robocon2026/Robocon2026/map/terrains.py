import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7,
            noise_range=(-0.05, 0.05),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)
