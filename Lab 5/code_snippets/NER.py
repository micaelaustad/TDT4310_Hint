from collections import Counter
from spacy import displacy
import wikipedia
import spacy


ma = wikipedia.WikipediaPage(title='Marcus Aurelius').summary
# python3 -m spacy download en_core_web_trf
nlp = spacy.load("en_core_web_trf")
doc = nlp(ma)

ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
print(ents)

"""
Marcus Aurelius Antoninus 0 25 PERSON
Latin 42 47 LANGUAGE
26 April 121 – 92 106 DATE
Roman 125 130 NORP
from 161 to 180 139 154 DATE
some 13 centuries later 257 280 DATE
Niccolò Machiavelli 284 303 PERSON
the Pax Romana 330 344 EVENT
27 BC to 180 346 358 DATE
the Roman Empire 404 420 GPE
Roman 435 440 NORP
140 451 454 DATE
145 456 459 DATE
161 465 468 DATE
Marcus 470 476 PERSON
Hadrian 506 513 PERSON
Marcus Annius Verus 551 570 PERSON
Domitia Calvilla 588 604 PERSON
three 634 639 DATE
Marcus 679 685 PERSON
Hadrian 693 700 PERSON
Aelius Caesar 717 730 PERSON
138 740 743 DATE
Marcus 765 771 PERSON
Antoninus Pius 779 793 PERSON
Antoninus 820 829 PERSON
Hadrian 876 883 PERSON
that year 889 898 DATE
Antoninus 903 912 PERSON
Marcus 953 959 PERSON
Greek 968 973 LANGUAGE
Latin 978 983 LANGUAGE
Marcus Cornelius Fronto 1025 1048 PERSON
Fronto 1087 1093 PERSON
many years 1098 1108 DATE
Marcus 1120 1126 PERSON
Antoninus 1135 1144 PERSON
Faustina 1155 1163 PERSON
145 1167 1170 DATE
Antoninus 1178 1187 PERSON
161 1196 1199 DATE
Marcus 1201 1207 PERSON
Lucius Verus 1280 1292 PERSON
Marcus Aurelius 1308 1323 PERSON
the Roman Empire 1370 1386 GPE
Kingdom of Armenia 1456 1474 GPE
Marcus 1476 1482 PERSON
Marcomanni 1496 1506 NORP
Quadi 1508 1513 NORP
Sarmatian Iazyges 1519 1536 NORP
the Marcomannic Wars 1540 1560 EVENT
Germanic 1587 1595 NORP
Empire 1651 1657 GPE
Roman 1696 1701 NORP
Christians 1745 1755 NORP
the Roman Empire 1759 1775 GPE
Marcus 1809 1815 PERSON
The Antonine Plague 1864 1883 EVENT
165 1897 1900 DATE
166 1904 1907 DATE
the Roman Empire 1941 1957 GPE
five million 1981 1993 CARDINAL
Lucius Verus 2002 2014 PERSON
169 2048 2051 DATE
Marcus 2086 2092 PERSON
Lucilla 2143 2150 PERSON
Lucius 2164 2170 PERSON
Commodus 2176 2184 PERSON
Marcus 2209 2215 PERSON
Marcus Aurelius 2328 2343 PERSON
Rome 2359 2363 GPE
Marcus 2515 2521 PERSON
Stoic 2587 2592 NORP
centuries 2687 2696 DATE
"""
