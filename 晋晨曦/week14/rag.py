
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import  InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import dashscope
from http import HTTPStatus
import time
class MeiTuanRag:
    def __init__(self):
        self.main()

    def main(self):
        self.compentents_loader()
        self.cutdocs()
        self.embed()
        self.revetrial()
        self.convertion()

    def compentents_loader(self):
        loader=TextLoader("knowledge.txt",encoding="utf-8")
        self.documents=loader.load()
        self.llm=ChatOpenAI(
            model="glm4:9b",
            base_url="http://localhost:11434/v1",  # 智谱API兼容地址
            openai_api_key="1234",
            stream=True
        )
        # self.llm=ChatOpenAI(
        #     api_key=os.environ['zhipuai_api_key'],
        #     base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱API兼容地址
        #     model="glm-4",
        #     stream=True
        # )
        self.embeddings=OpenAIEmbeddings(
            api_key=os.environ['zhipuai_api_key'],
            base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱API兼容地址
            model="embedding-2"
        )
        self.prompt=ChatPromptTemplate.from_messages([
            ("system","你是一名美团外卖app智能客服,需要结合历史对话和检索到的知识库回答,\n知识：{context}\n历史:{history},请你直接返回和我的问题对应的知识库中的整条答案,不要删减或自己加东西,如果有不太清楚但是知识库比较相关的内容，直接用知识库内容就好,如果有一个问题对应多种情况的答案,没有特殊说明的话输出多种情况,如果只问到某一种或者某几种答案的话,输出对应几种情况的答案,如果完全不相关，输出我不知道"),
            ("user","{input}")
        ])
        self.history=InMemoryChatMessageHistory()

    def cutdocs(self):
        flag=0
        if flag==0:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,  # 适当增大片段长度（根据知识库内容调整）
                chunk_overlap=0,  # 保留部分重叠，避免切割语义
                # separators=["\n\n", "\n", "。", "，", " "]  # 按优先级分割，更符合中文习惯
            )
            self.segements = text_splitter.split_documents(self.documents)
        #文本分割
        else:
            text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=200,chunk_overlap=0)
            self.segements=text_splitter.split_documents(self.documents)
        texts=[chunk.page_content for chunk in self.segements]
        # print(f"分割后的片段长度为{len(texts)}")
        # for document in texts:
        #     print(document)
        #     print("--------------------------------")

    def embed(self):
        # 正确步骤：先删除旧集合，再创建新集合，最后查看内容
        from chromadb import PersistentClient
        # 1. 连接到持久化目录并删除旧集合（关键：在创建新集合前执行）
        client = PersistentClient(path="../chroma.db")
        if "langchain" in [col.name for col in client.list_collections()]:
            client.delete_collection(name="langchain")
            # print("已删除旧集合")

        # 2. 重新创建向量库（此时集合为空，只会添加当前的segements）
        self.vector = Chroma.from_documents(
            self.segements,
            self.embeddings,
            persist_directory="../chroma.db"
        )
        # print(f"已添加新片段，数量：{len(self.segements)}")

        # 3. 查看向量库内容
        client = PersistentClient(path="../chroma.db")
        collection = client.get_collection(name="langchain")  # 获取刚创建的集合
        all_docs = collection.get(limit=100)
        # print(f"向量库中实际存储的片段数量: {len(all_docs['ids'])}")
        # for i in range(len(all_docs['ids'])):
        #     print(f"\n片段ID: {all_docs['ids'][i]}")
        #     print(f"内容: {all_docs['documents'][i]}")
        #     print(f"元数据: {all_docs['metadatas'][i]}")
        #     print("-" * 50)

    def revetrial(self):
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval = self.vector.as_retriever(
            search_type="mmr",  # 最大化边际相关性
            search_kwargs={
                "k": 15,
                "lambda_mult": 0.1  # 平衡相关性和多样性
            }
        )

    def text_rerank(self,query, candidate_docs):
        # 提取候选文档内容
        docs = [d.page_content for d in candidate_docs]
        # 调用重排序API
        resp = dashscope.TextReRank.call(
            model="gte-rerank-v2",
            query=query,
            documents=docs,
            top_n=6,
            api_key=os.environ.get("aliyunbailian_api_key")
        )

        # 强制用候选文档内容兜底（忽略API返回的null）
        if resp.status_code == HTTPStatus.OK and "results" in resp.output:
            reranked = []
            for item in resp.output["results"]:
                # 直接使用候选文档的原始内容（因为API返回null）
                original_doc = candidate_docs[item["index"]]
                reranked.append(Document(
                    page_content=original_doc.page_content,  # 用候选文档内容
                    metadata=original_doc.metadata
                ))
            return reranked[:6]
        else:
            # API调用失败时，直接返回前3个候选文档
            return [Document(page_content=d.page_content, metadata=d.metadata) for d in candidate_docs[:3]]

    def safe_stream_generate(self, input_data, max_retries=3):
        retry_count = 0
        chunk_index = 0  # 序号标记，保证顺序
        first_try=True
        while retry_count < max_retries:
            try:
                stream = self.document_chain.stream(input_data)
                for chunk in stream:
                    if first_try and chunk_index == 3:  # 生成到第2个chunk时故意报错
                        first_try = False
                        raise Exception("模拟网络中断")
                    yield (chunk_index, chunk)  # 返回(序号, 内容)
                    chunk_index += 1
                break
            except Exception as e:
                retry_count += 1
                print(f"重试({retry_count}/{max_retries})：{e}")
                if retry_count >= max_retries:
                    yield (-1, "[内容传输中断，请重试]")  # 失败标记
                time.sleep(1)

    def convertion(self):
        while True:
            user_input = input("请提出你的问题:")
            print("思考中-------")
            if not user_input:
                print("输入不能为空，请重新输入。")
                continue  # 空输入时跳过后续处理，直接进入下一轮

            #粗排
            start_time=time.time()
            candidate_docs=self.retrieval.invoke(user_input)
            middle_time=time.time()
            flag=1
            if flag==0:
                for index,candidate_doc in enumerate(candidate_docs):
                    print(f"粗排第{index+1}条")
                    print(candidate_doc.page_content)
                    print("___________________")
            # print(f"粗排耗时{round((middle_time-start_time)/60,2)}")

            #精排
            middle_time=time.time()
            reranked_docs=self.text_rerank(user_input,candidate_docs)
            end_time=time.time()
            if flag==0:
                for index,text in enumerate([doc.page_content for doc in reranked_docs]):
                    print(f"重排序后的第{index+1}条:")
                    print(text)
                    print("______")
            # print(f"精排耗时{round((end_time-middle_time)/60,2)}")

            #执行链条，流式输出
            input_data={"input":user_input,
                        "history":self.history,
                        "context":reranked_docs
                        }
            # response = self.document_chain.stream(input_data)
            response_full = ""
            print("回答：", end="", flush=True)
            for chunk_index, chunk in self.safe_stream_generate(input_data):
                if chunk_index == -1:
                    print(chunk, end="", flush=True)  # 输出错误提示
                else:
                    print(chunk, end="", flush=True)  # 正常输出内容
                response_full += chunk
            print("")
            #结束问答，保存该轮对话内容
            # print("回答完毕")
            self.history.add_message(HumanMessage(content=user_input))
            self.history.add_message(AIMessage(content=response_full))

if __name__ == "__main__":
    rag=MeiTuanRag()
    # print(len([chunk.page_content for chunk in self.segements]))


以下是美团常见问题风格编写的完整问答对
### 在线支付问题
Q：在线支付取消订单后钱怎么返还？
A：若商家未接单，取消订单后金额会自动退回支付账户（如原银行卡、美团余额等，依支付方式而定 ）；若商家已接单，需在订单页提交退款申请，商家审核通过后，款项按原支付路径返还，一般1 - 3个工作日到账（不同支付渠道到账时间有差异 ）。

Q：怎么查看退款是否成功？
A：打开美团外卖 APP，进入“我的” - “我的订单”，找到对应订单查看退款状态；也可查看支付账户（如银行卡账单、微信/支付宝账单 ），确认款项是否退回。

Q：美团账户里的余额怎么提现？
A：美团外卖余额暂不支持直接提现。可用于美团外卖下单支付、购买美团平台支持余额消费的商品/服务，后续若有提现功能调整，以美团官方公告为准 。

Q：余额提现到账时间是多久？
A：美团外卖余额无法直接提现，若因退款等场景退回余额，余额到账无延迟（退款审核通过后即时到账美团余额 ），使用余额支付下单也无到账等待期 。

Q：申请退款后，商家拒绝了怎么办？
A：可在订单退款详情页查看商家拒绝理由，若理由不成立，点击“申请平台介入”，上传订单截图、沟通凭证等资料，美团客服会核实处理，一般1 - 2个工作日给出判定结果 。

Q：怎么取消退款呢？
A：在美团外卖 APP 订单退款流程中，若商家未处理退款申请，可进入“我的订单” - 对应订单退款页，点击“取消退款”；若商家已处理（同意或拒绝 ），则无法取消，需重新下单 。

Q：前面下了一个在线支付的单子，由于未付款，订单自动取消了，这单会计算我的参与活动次数吗？
A：未付款且订单自动取消，不算成功参与活动，不会计入活动次数。活动参与以“成功支付下单”为判定标准，仅创建订单未付款不满足活动参与条件 。

Q：为什么我用微信订餐，却无法使用在线支付？
A：可能是微信账号未绑定有效支付方式、支付限额超限，或美团与微信支付临时系统兼容问题。可检查微信支付设置，更换网络重新尝试，或联系美团客服、微信支付客服排查 。

Q：如何进行付款？
A：下单选好商品后，进入结算页，选择在线支付方式（如微信支付、支付宝、美团支付等 ），点击“提交订单”并完成支付验证（输入密码、指纹、人脸等 ），即可完成付款 。

Q：如何查看可以在线支付的商家？
A：下单时，商家列表无特殊标识区分是否支持在线支付，选好商品进入结算页，若有微信支付、支付宝等在线支付选项，即代表该商家支持在线支付；若仅有“到店付款”等离线选项，则暂不支持 。

Q：美团外卖支持哪些支付方式？
A：支持微信支付、支付宝、美团支付（余额/银行卡绑定 ）、银联在线支付等，部分商家还支持Apple Pay，下单结算页可查看当前订单可用支付方式 。

Q：在线支付订单如何退款？
A：商家接单前，取消订单可直接退回款项；商家接单后，在订单页点“申请退款”，选退款原因并提交，商家24小时内处理，同意退款则款项按原路径返还；若商家超时未处理，系统自动同意退款 。

Q：在线支付的过程中，订单显示未支付成功，款项却被扣了，怎么办？
A：先等待10 - 30分钟，可能是支付系统延迟，美团订单状态会自动同步。若仍显示未支付，保留支付成功凭证（如银行卡扣款短信、支付平台账单 ），联系美团客服10109777，说明订单号、支付情况，客服会核实处理，一般1 - 2个工作日恢复订单或退款 。


### 优惠问题
Q：哪些商家有优惠？都有些什么优惠？
A：打开美团外卖 APP，进入商家列表，带有“优惠”“满减”“折扣”等标签的商家即为有优惠。优惠类型包括满减（如满30减5 ）、折扣菜（特定菜品低价 ）、新客立减、买赠（如买饭送饮料 ）等，具体优惠点进商家主页查看“店铺优惠”板块 。

Q：在新用户享受的优惠中，新用户的条件是什么？
A：美团外卖新用户指从未在美团外卖下单的用户，同一手机号、设备、美团账号，满足任一即为“老用户”，仅新用户可享受首单优惠（如新客立减、新客专属折扣 ） 。

Q：我达到了满赠、满减的优惠的金额，为什么没有享受相关的优惠？
A：先检查优惠规则，满赠/满减是否限商品品类（如仅菜品参与，配送费、包装费不计入 ）、是否限特定时段/新老客；若规则符合仍未享受，进入订单结算页，确认是否勾选对应优惠，或联系商家、美团客服核实 。

Q：超时赔付是什么意思？
A：商家承诺订单送达时间，若实际送达时间超过承诺时间（恶劣天气、商家出餐慢等合理延误除外 ），用户可获得赔付。赔付形式一般为美团红包（可用于下次下单抵扣 ），具体赔付规则以商家详情页“超时赔付”说明为准 。


### 订单问题
Q：为什么提示我“账户存在异常，无法下单”？
A：包含（但不仅限于）以下行为者，系统将自动予以封禁（客服无权解封）：
i）有过虚假交易（编造不存在真实买卖的订单）；
ii）有过恶意下单行为（如频繁取消订单、下单后拒接配送电话影响履约 ）；
iii）账号安全存在风险（如异地异常登录、密码泄露 ）。可联系美团客服排查具体原因 。

Q：如何取消订单？
A：商家未接单时，进入“我的订单” - 对应订单，点击“取消订单”，选取消原因即可；商家已接单，需联系商家协商取消，协商一致后，商家操作取消订单，或联系美团客服协助沟通 。

Q：我的订单为什么被取消了？
A：可能是商家原因（如商品售罄、营业时间外下单 ），商家会电话/短信告知；也可能是系统判定订单异常（如支付未完成、账号风险 ）。可在“我的订单”查看取消原因，或联系商家、美团客服核实 。

Q：如何进行催单？
A：打开美团外卖 APP，进入“我的订单”，找到对应订单，点击“催单”按钮，系统会自动给商家发送催单提醒；也可电话联系商家，说明订单号催促出餐/配送 。

Q：刚下单发现信息填错了怎么办？
A：若商家未接单，进入订单页点击“修改订单”，可修改收货地址、联系电话等信息（部分信息如商品品类无法修改，需取消重下 ）；若商家已接单，联系商家说明情况，协商能否修改，或取消订单重新下单 。

Q：我的订单是否被商家确认？
A：进入美团外卖“我的订单”，若订单显示“商家已接单”，则代表商家确认；若显示“待商家接单”，则还未确认。也可电话联系商家，报订单号核实 。

Q：预计送达的时间为什么与我实际收餐的时间不符？
A：受商家出餐速度（如订单高峰期出餐慢 ）、配送距离（路况复杂、偏远地址 ）、天气因素（雨雪天骑手配送慢 ）影响，可能导致实际收餐时间延迟。若延迟严重，可联系骑手或商家了解进度，或联系美团客服反馈 。

Q：为什么会出现无法下单的情况？
A：可能是收货地址超出配送范围（商家不配送该区域 ）、所选商品库存不足、账号异常（如被封禁 ）、支付方式故障（余额不足、支付限额 ）等。可检查地址、商品库存，尝试更换支付方式，或联系美团客服排查 。

Q：为什么提示下单次数过多，已无法下单？
A：短时间内频繁下单（如1小时内下单超5单 ），系统会判定为异常操作限制下单，旨在防范恶意下单行为。可等待1 - 2小时后再尝试，或联系美团客服说明合理需求（如批量下单采购 ），申请解除限制 。


### 其他问题
Q：如果对商家服务不满意如何进行投诉？
A：进入美团外卖“我的订单”，找到对应订单，点击“投诉商家”，选择投诉类型（如服务态度差、菜品质量问题 ），上传凭证（照片、聊天记录等 ）提交；也可联系美团客服，说明情况协助投诉处理 。

Q：如何联系客服解决问题？
A：打开美团外卖 APP，进入“我的” - “客服中心”，可选择“在线客服”实时咨询；或拨打美团外卖客服电话10109777，按语音提示选择对应服务（如订单问题、投诉建议 ） 。

Q：我用的是手机客户端，为什么无法定位？
A：检查手机设置，确保美团外卖 APP 开启“位置权限”；尝试刷新页面，或退出 APP 重新进入；若仍无法定位，切换网络（如从4G切为 Wi - Fi ），或联系手机厂商客服排查定位功能故障 。

Q：如何修改自己的账户信息？
A：打开美团 APP（美团外卖账户信息与美团主站通用 ），进入“我的” - “设置” - “个人信息”，可修改昵称、头像、联系电话、收货地址等信息，部分核心信息（如实名认证 ）修改需严格验证 。

Q：为什么有时需要输入短信验证码？
A：涉及账号安全操作（如修改登录密码、更换绑定手机、异地大额支付 ）时，系统会发送短信验证码到绑定手机，验证是本人操作，保障账号资金安全 。
