import numpy as np
from py2neo import Graph

graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

#手动创建neofj的实体、关系、属性
def creat_graph():
    cypher = ""
    # 创建5个叉车类型 【实体】
    cypher += "CREATE (搬马机器人:叉车类型 {name:'搬马机器人', 供能:'锂电'}) \n"
    cypher += "CREATE (电动拣选车:叉车类型 {name:'电动拣选车', 供能:'锂电'}) \n"
    cypher += "CREATE (电动搬运车:叉车类型 {name:'电动搬运车', 供能:'锂电'}) \n"
    cypher += "CREATE (电动堆高车:叉车类型 {name:'电动堆高车', 供能:'锂电'}) \n"
    cypher += "CREATE (内燃叉车:叉车类型 {name:'内燃叉车', 供能:'燃油'}) \n"

    # 给”搬马机器人“创建3个品名 【实体】
    cypher += "CREATE (XCD101:叉车 {name:'XCD101', 额定载荷:'1000', 操作类型:'手柄遥控', 全名:'XCD101潜伏顶升搬运机器人'}) \n"
    cypher += "CREATE (XCD021:叉车 {name:'XCD021', 额定载荷:'200', 操作类型:'手柄遥控', 全名:'XCD021 点对点顶升搬运机器人'}) \n"
    cypher += "CREATE (XCD051:叉车 {name:'XCD051', 额定载荷:'500', 操作类型:'手柄遥控', 全名:'XCD051 点对点顶升搬运机器人'}) \n"
    # 给”电动拣选车“创建3个品名 【实体】
    cypher += "CREATE (CQD14:叉车 {name:'CQD14', 额定载荷:'1400', 操作类型:'操纵杆', 全名:'CQD14 1.4吨锂电前移式叉车'}) \n"
    cypher += "CREATE (RQL151:叉车 {name:'RQL151', 额定载荷:'1500', 操作类型:'手柄遥控', 全名:'RQL151 1.5吨前移式电动叉车'}) \n"
    cypher += "CREATE (CQD18S2:叉车 {name:'CQD18S2', 额定载荷:'1800', 操作类型:'操纵杆', 全名:'CQD18S2 1.8吨站驾前移式电动叉车'}) \n"

    # 给”电动搬运车“创建1个品名 【实体】
    cypher += "CREATE (EPT20:叉车 {name:'EPT20', 额定载荷:'1400', 操作类型:'操纵杆', 全名:'EPT20 1.5吨电动搬运车'}) \n"
    # 给”电动堆高车“创建1个品名 【实体】
    cypher += "CREATE (ES10:叉车 {name:'ES10', 额定载荷:'1000', 操作类型:'操纵杆', 全名:'ES10 1.2吨电动堆高车(单级窄腿)'}) \n"
    # 给”内燃叉车“创建1个品名 【实体】
    cypher += "CREATE (CPC20T3:叉车 {name:'CPC20T3', 额定载荷:'1400', 操作类型:'操纵杆', 全名:'CPC20T3 2.0吨内燃叉车'}) \n"

    #简历实体之间的关联关系 【关系】
    cypher += "CREATE (XCD101)-[:品名]->(搬马机器人) \n"
    cypher += "CREATE (XCD021)-[:品名]->(搬马机器人) \n"
    cypher += "CREATE (XCD051)-[:品名]->(搬马机器人) \n"

    cypher += "CREATE (CQD14)-[:品名]->(电动拣选车) \n"
    cypher += "CREATE (RQL151)-[:品名]->(电动拣选车) \n"
    cypher += "CREATE (CQD18S2)-[:品名]->(电动拣选车) \n"

    cypher += "CREATE (EPT20)-[:品名]->(电动搬运车) \n"
    cypher += "CREATE (ES10)-[:品名]->(电动堆高车) \n"
    cypher += "CREATE (CPC20T3)-[:品名]->(内燃叉车) \n"

    print(cypher)
    # 执行建表脚本
    graph.run(cypher)

#根据节点名称删除名下所有标签
def delete_graph_by_label_name(label_name):
    cypher = f"""
        MATCH (n:{label_name}) 
        DETACH DELETE n
    """
    print("根据节点名称删除名下所有标签：\n", cypher)
    graph.run(cypher)


if __name__ == '__main__':
    # True False
    is_create = False
    is_delete = False

    #手动创建neofj的实体、关系、属性
    if is_create:
        creat_graph()

    # 根据标签名称删除名下所有标签
    if is_delete:
        delete_graph_by_label_name("叉车类型")
        delete_graph_by_label_name("叉车")

    cypher = "Match (n) where n.name='搬马机器人' return n.供能"
    data = graph.run(cypher).data()
    print("data：\n", data)
