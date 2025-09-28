import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TERRAINS_CFG = TerrainGeneratorCfg(
    # 每个地形块大小为8x8米
    size=(8.0, 8.0),
    # 边界宽度20米
    border_width=20.0,
    # 地形网格为10行20列
    num_rows=10,
    num_cols=20,
    # 水平和垂直分辨率
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # 包含两种子地形
    sub_terrains={
        # 平坦地形，占比30%
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        # 随机粗糙地形，占比70%，有-0.05到0.05米的噪声范围
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7,
            noise_range=(-0.05, 0.05),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)
