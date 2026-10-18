from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app.py", default_timeout=120)
at.run()
assert not at.exception, at.exception
print("step0 buttons:", [b.label for b in at.button])
at.button[0].click().run()          # Далее
print("step1 buttons:", [b.label for b in at.button])
demo = [b for b in at.button if "демо" in b.label][0]
demo.click().run()
assert not at.exception, at.exception
print("step2 title:", [t.value for t in at.title])
print("metrics:", [(m.label, m.value) for m in at.metric])
print("questions:", len(at.markdown))
chek = [b for b in at.button if "демо-чеки" in b.label][0]
chek.click().run()
assert not at.exception, at.exception
print("after receipts:", [(m.label, m.value) for m in at.metric])
[b for b in at.button if "итоговому" in b.label][0].click().run()
assert not at.exception, at.exception
print("step3:", [(m.label, m.value) for m in at.metric])
