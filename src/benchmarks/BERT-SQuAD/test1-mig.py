from bert import QA
from timeit import default_timer as timer

IS_COLD = True
model = None

def init():
      global model
      # before = timer()
      model = QA('/users/pzhou/projects/gpuless/src/benchmarks/BERT-SQuAD/model')
      # after = timer()

      # print('model eval time:')
      # print(after - before)

def inference(doc, q):
      start = timer()

      answer = model.predict(doc, q)
      # print(answer['answer'])
      # print(answer.keys())

      end = timer()
      # print(end - start)


def client_main():
      doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
            "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
            "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
            "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
            "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
            "people in a referendum, depending on the provision. "

      q = 'When did Victoria enact its constitution?'

      for i in range(101):
            global IS_COLD
            start = timer()
            if IS_COLD:
                  init()
                  IS_COLD = False
            inference(doc, q)
            end = timer()
            
            print(end - start)
            
if __name__ == "__main__":
    client_main()