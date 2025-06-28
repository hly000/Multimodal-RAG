import pandas as pd

# 模拟的风控审核数据（使用100%来自Wikimedia Commons的超稳定链接）
data = {
    "url": [
        "https://upload.wikimedia.org/wikipedia/commons/f/f3/Beretta_92FS_-_USA.jpg",                                            # Handgun
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Rolex_Submariner_116610LV.jpg/1200px-Rolex_Submariner_116610LV.jpg", # "Counterfeit" watch
        "https://upload.wikimedia.org/wikipedia/commons/3/3f/BM42S.jpg",                                                        # Tactical knife
        "https://upload.wikimedia.org/wikipedia/commons/8/87/Chefs-knife.jpg",                                                  # Kitchen knife
        "https://upload.wikimedia.org/wikipedia/commons/a/a2/Super_Soaker_50_20th_Anniversary_Edition_-_Soak-a-thon_2009.jpg",      # Water gun
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Gucci_bag.jpg/1024px-Gucci_bag.jpg",                            # "Counterfeit" bag
        "https://upload.wikimedia.org/wikipedia/commons/7/7b/Kitchen-spatula.jpg",                                              # Kitchen spatula
        "https://upload.wikimedia.org/wikipedia/commons/a/a3/Slingshot-pro.jpg",                                                # Slingshot
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Foam_sword.jpg/1200px-Foam_sword.jpg",                        # Toy sword
        "https://upload.wikimedia.org/wikipedia/commons/1/1a/Zippo-Street-Chrome-207.jpg"                                       # Zippo lighter
    ],
    "desc": [
        "疑似制式手枪。根据平台规则，任何制式枪械及其主要部件均属于一级违禁品，严禁销售。",
        "疑似劳力士'水鬼'高仿手表。表盘细节、表冠护肩与正品有差异，根据打假规则，判定为高风险仿冒品。",
        "带有锁定装置的蝴蝶刀。根据平台规则，判定为管制刀具，属于一级违禁品。",
        "标准西式主厨刀。刀具本身无锁定装置，符合厨房用具规范，允许销售。需提醒商家在详情页标明'厨房专用'。",
        "儿童玩具水枪，色彩鲜艳，外形非仿真。根据平台规则，判定为安全玩具，允许销售。",
        "古驰（Gucci）品牌手提包，但其Logo、缝线及五金件均与正品不符。根据打假规则，判定为高风险仿冒品。",
        "普通厨房用锅铲。无任何危险性，材质为不锈钢和硅胶，属于普通餐厨用具，允许销售。",
        "弹弓，配有腕托和激光瞄准器。根据平台规则，此类具有较强杀伤力的弹弓属于三级违禁品，禁止销售。",
        "儿童玩具泡沫剑。材质为软性泡沫，无锋利边缘，符合儿童玩具安全标准，允许销售。",
        "Zippo品牌金属煤油打火机。根据平台规则，打火机属于受限制商品，允许销售，但必须通过指定的陆运物流发货。"
    ],
    "category": [
        "risk/prohibited/firearms",
        "risk/prohibited/counterfeit_luxury",
        "risk/prohibited/controlled_knives",
        "risk/allowed/kitchenware_knife",
        "risk/allowed/toy_gun",
        "risk/prohibited/counterfeit_luxury",
        "risk/allowed/kitchenware",
        "risk/prohibited/slingshot_weapon",
        "risk/allowed/toy_sword",
        "risk/restricted/lighters"
    ]
}

df = pd.DataFrame(data)

# 将DataFrame保存为Excel文件
df.to_excel("dataset/template.xlsx", index=False) 