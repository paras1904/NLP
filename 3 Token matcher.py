import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

pattern1 = [{'LOWER':'solarpower'}]
pattern2 = [{'LOWER':'solar'},{'LOWER':'power'}]
pattern3 = [{'LOWER':'solar'},{'IS_PUNCT':True},{'LOWER':'power'}]
matcher.add('solarPOwer',None,pattern1,pattern2,pattern3)

doc = nlp(u'the Solar Power industry continues to grow as a demand for solarpower increases. solar-power cars are gaining popularity.')
found_matches = matcher(doc)
print(found_matches)
for match_id,start,end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id,string_id,start,end,span.text)

print()

# op function addition incllude !?-+
# adding lemma
# you can add more patterns so that it read the following word
matcher2 = Matcher(nlp.vocab)
pattern01 = [{'LOWER':'solarpower'}]
pattern02 = [{'LOWER':'solar'},{'IS_PUNCT': True,'OP':'*'},{'LEMMA':'power'}]

matcher2.add('SolarPower',None,pattern01,pattern02)

doc2 = nlp(u'Solar-powered energy runs solar-powered cars.')
found_matches2 = matcher2(doc2)
print(found_matches2)
for match_id2, start2,end2 in found_matches2:
    string_id2 = nlp.vocab.strings[match_id2]
    span2 = doc[start2:end2]
    print(match_id2,start2,end2,span2.text)
