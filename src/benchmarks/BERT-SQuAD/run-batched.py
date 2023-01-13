from bert import QA
from timeit import default_timer as timer

before = timer()
model = QA('model')
after = timer()

print('model eval time:')
print(after - before)


docs = [
    "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the "
    "United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's "
    "law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the "
    "Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in "
    "both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, "
    "depending on the provision.",
    'At the 52nd Annual Grammy Awards, Beyoncé received ten nominations, including Album of the Year for I Am... '
    'Sasha Fierce, Record of the Year for "Halo", and Song of the Year for "Single Ladies (Put a Ring on It)", '
    'among others. She tied with Lauryn Hill for most Grammy nominations in a single year by a female artist. In '
    '2010, Beyoncé was featured on Lady Gaga\'s single "Telephone" and its music video. The song topped the US Pop '
    'Songs chart, becoming the sixth number-one for both Beyoncé and Gaga, tying them with Mariah Carey for most '
    'number-ones since the Nielsen Top 40 airplay chart launched in 1992. "Telephone" received a Grammy Award '
    'nomination for Best Pop Collaboration with Vocals. '
]

qs = [
    'When did Victoria enact its constitution?',
    'How many awards was Beyonce nominated for at the 52nd Grammy Awards?',
]

total_time = 0
times = []

# warmup
for i in range(0, 5):
    model.predict(docs[i % 2], qs[i % 2])

iterations = 100
# benchmark
for i in range(0, iterations):
    start = timer()
    model.predict(docs[i % 2], qs[i % 2])
    end = timer()
    times.append(end - start)

total_time = sum(times) * 1000
avg_time = total_time / float(iterations)
print("Avg time: {:}ms".format(round(avg_time, 5)))
