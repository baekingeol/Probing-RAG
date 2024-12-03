def prompt(query):
    qa_query = f"<s>[INST] Answer the following question. Just answer in 5 words or less. [/INST] \nQ: {query} \nA:"
    return qa_query

def icl_prompt(q):
    qa_query = f'''Example 1:
Question: What is the capital of France?
Answer: Paris

Example 2:
Question: Who wrote the play "Romeo and Juliet"?
Answer: William Shakespeare

Example 3:
Question: Which planet is known as the Red Planet?
Answer: Mars

Example 4:
Question: What is the largest ocean on Earth?
Answer: Pacific Ocean

Example 5:
Question: Who painted the Mona Lisa?
Answer: Leonardo da Vinci

Example 6:
Question: {q}
Answer:'''
    return qa_query
def inst_prompt(q):
    qa_query = f'''Instruction: Answer the following question. Just answer in 5 words or less. Don't generate any additional comments, just write your answer.

Question: 
{q}

Answer:'''
    return qa_query


def cot_prompt_old(query):
    q = f'''Query: Who was the first President of the United States?
Rationale: The United States was formed after gaining independence from Britain. The first President would have been elected soon after the formation of the country. George Washington is commonly known as the leader of the American Revolution and the first President.
Answer: George Washington

Query: Who wrote the play 'Romeo and Juliet'?
Rationale: “Romeo and Juliet” is a famous play, a tragedy involving two young lovers. The play is widely associated with English literature from the Renaissance period. William Shakespeare is the most renowned playwright of the English Renaissance and is known for his tragedies.
Answer: William Shakespeare

Query: What is the main ingredient in traditional Italian pesto sauce?
Rationale: Pesto is a famous Italian sauce typically used in pasta dishes. Traditional pesto originates from the Liguria region, specifically Genoa. The primary ingredients include basil, olive oil, garlic, pine nuts, and Parmesan cheese. The main flavoring herb that distinguishes pesto is basil.
Answer: Basil

Query: Who wrote the play 'Romeo and Juliet'?
Rationale: “Romeo and Juliet” is a famous play, a tragedy involving two young lovers. The play is widely associated with English literature from the Renaissance period. William Shakespeare is the most renowned playwright of the English Renaissance and is known for his tragedies.
Answer: William Shakespeare

Query: {query}
Rationale:'''
    return q 

def cot_prompt(query):
    q = f'''Question: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
Rationale: Evidence 2 identifies 2014 S/S as the debut album of the South Korean group WINNER and states that it was released by YG Entertainment. This directly connects the group WINNER with YG Entertainment.
Answer: WINNER

Question: Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?
Rationale: Roger O. Egeberg served in this role during the Nixon administration. It also specifies other roles he held, but the key information is his service under President Nixon. To determine the years, we need to know the timeframe of Nixon's presidency.
Answer: 1969 to 1974

Question: Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?
Rationale: Ambroise Thomas (1811-1896) is a French opera composer, listing his complete operas. Giuseppe Verdi (1813-1901) was an Italian opera composer, detailing his work and lifespan.
Answer: Yes

Question: Which writer was from England, Henry Roth or Robert Erskine Childers?
Rationale: Henry Roth was an American novelist, best known for his novel Call It Sleep (1934). He was born in Austria-Hungary (now Ukraine) and emigrated to the United States as a child. Robert Erskine Childers was an English-born writer and Irish nationalist. He was born in London, England, in 1870 and is best known for his novel The Riddle of the Sands (1903). Childers later became involved in Irish politics and was a prominent figure in the Irish independence movement.
Answer: Robert Erskine 

Question: {query}
Rationale:'''
    return q

def retr_qa(query, docs: str):
    q = f'''[INST]Instruction: Write an Answer to the Query based on the Query and Evidence. Just answer in 5 words or less. Don't generate any additional comments, just write your answer.[/INST]
Question: 
{query}

Passages: 
{docs}

Answer:'''
    return q

def retr_qa_cot2(query, docs: str):
    q = f'''Query: Who was the first President of the United States?
Rationale: The United States was formed after gaining independence from Britain. The first President would have been elected soon after the formation of the country. George Washington is commonly known as the leader of the American Revolution and the first President.
Answer: George Washington

Query: Who wrote the play 'Romeo and Juliet'?
Rationale: “Romeo and Juliet” is a famous play, a tragedy involving two young lovers. The play is widely associated with English literature from the Renaissance period. William Shakespeare is the most renowned playwright of the English Renaissance and is known for his tragedies.
Answer: William Shakespeare

Query: What is the main ingredient in traditional Italian pesto sauce?
Rationale: Pesto is a famous Italian sauce typically used in pasta dishes. Traditional pesto originates from the Liguria region, specifically Genoa. The primary ingredients include basil, olive oil, garlic, pine nuts, and Parmesan cheese. The main flavoring herb that distinguishes pesto is basil.
Answer: Basil

Question: Which writer was from England, Henry Roth or Robert Erskine Childers?
Rationale: Henry Roth was an American novelist, best known for his novel Call It Sleep (1934). He was born in Austria-Hungary (now Ukraine) and emigrated to the United States as a child. Robert Erskine Childers was an English-born writer and Irish nationalist. He was born in London, England, in 1870 and is best known for his novel The Riddle of the Sands (1903). Childers later became involved in Irish politics and was a prominent figure in the Irish independence movement.
Answer: Robert Erskine 

Passages: 
{docs}
Question: 
{query}
Rationale:'''
    return q

def retr_qa_cot(query, docs: str):
    q = f'''Question: 
2014 S/S is the debut album of a South Korean boy group that was formed by who?
Passages: 
passage 1: S (에스) is a South Korean project group consisting of three members: Kangta, Lee Ji-hoon and Shin Hye-sung. The group debuted in 2003, under the SM Entertainment label. After 11 years, they released and promoted another mini-album in 2014.
passage 2: 2014 S/S is the debut album of South Korean group WINNER. It was released on August 12, 2014 by the group's record label, YG Entertainment. The members were credited for writing the lyrics and composing the majority of the album's songs.
passage 3: The discography of South Korean girl group S.E.S. consists of seven studio albums, four compilation albums, three video albums, and one remix album. The group debuted in 1997 under SM Entertainment and disbanded in 2002. The group then made their comeback in 2016.
passage 4: The Wonder Years is the debut studio album by South Korean girl group Wonder Girls. It was released on September 13, 2007, by JYP Entertainment.
passage 5: Girls' Generation is the debut Japanese album by South Korean Girls' Generation, released by Nayutawave Records and Universal Music Japan on June 1, 2011. A repackaged edition of the album, "The Boys", was released on December 28, 2011.
Rationale: Evidence 2 identifies 2014 S/S as the debut album of the South Korean group WINNER and states that it was released by YG Entertainment. This directly connects the group WINNER with YG Entertainment.
Answer: WINNER

Question: 
Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?
Passages:
passage 1: Roger Olaf Egeberg, M.D. (13 November 1902 - 13 September 1997 Washington, D.C.) was an American medical educator, administrator and advocate of public health. He was General Douglas MacArthur's personal physician during World War II in the Pacific theater. His other roles included Assistant Secretary for Health and Scientific Affairs in the Department of Health, Education, and Welfare (now the United States Department of Health and Human Services) during the Nixon administration and Dean of the University of Southern California's medical school.
passage 2: Charles Edwin Johnson served as Acting United States Secretary of Health and Human Services from January to April 2009 during the Obama presidency. Johnson was appointed by then President George W. Bush in 2005 as Assistant Secretary for Budget, Technology, and Finance. He was a public accountant for 31 years prior to joining the department.
passage 3: United States Secretary of Health and Human Services
passage 4: United States Deputy Secretary of Health and Human Services
passage 5: Roger M. Natsuhara was the acting United States Assistant Secretary of the Navy (Installations and Environment).
Rationale: Roger O. Egeberg served in this role during the Nixon administration. It also specifies other roles he held, but the key information is his service under President Nixon. To determine the years, we need to know the timeframe of Nixon’s presidency.
Answer: 1969 to 1974

Question: 
Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?
Passages:
passage 1: This is a list of the complete operas of the French opera composer Ambroise Thomas (1811-1896). All premieres took place in Paris unless otherwise noted.
passage 2: Verdi refers to Giuseppe Verdi (1813-1901), Italian opera composer.
passage 3: The following is a list of published compositions by the composer Giuseppe Verdi (1813-1901).
passage 4: Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 - 27 January 1901) was an Italian opera composer.
passage 5: This is an incomplete list of authors who have written libretti for operas. Only librettists with their own articles in Wikipedia are listed. The name of the composer of each opera is also given.
Rationale: Ambroise Thomas (1811-1896) is a French opera composer, listing his complete operas. Giuseppe Verdi (1813-1901) was an Italian opera composer, detailing his work and lifespan.
Answer: Yes

Question: 
Which writer was from England, Henry Roth or Robert Erskine Childers?
Passages:
passage 1: Henry Roth was a writer.
passage 2: Henry Roth (February 8, 1906 - October 13, 1995) was an American novelist and short story writer.
passage 3: This is a list of novelists from England.
passage 4: Philip Milton Roth (born March 19, 1933) is an American novelist.
passage 5: Henry Melville was an Australian writer.
Rationale: Henry Roth was an American novelist, best known for his novel Call It Sleep (1934). He was born in Austria-Hungary (now Ukraine) and emigrated to the United States as a child. Robert Erskine Childers was an English-born writer and Irish nationalist. He was born in London, England, in 1870 and is best known for his novel The Riddle of the Sands (1903). Childers later became involved in Irish politics and was a prominent figure in the Irish independence movement.
Answer: Robert Erskine 

Question: 
{query}
Passages: 
{docs}
Rationale:'''
    return q

def dummy_prompt(query):
    return query

def flare_query_reformulation_prompt(query, masked_text):
    reformulated_query = f'''User input: Generate a summary about Joe Biden
Generated output so far: Joe Biden attended ____, where he earned a law degree.
Given the above passage, ask a question to which the answer is the term/entity/phrase: What university did Joe Biden attend?

User input: {query}
Generated output so far: {masked_text}
Given the above passage, ask a question to which the answer is the term/entity/phrase:'''
    return reformulated_query

def flare_for_new_sent_prompt(y_t, evidences):
    for_make_new_sent=f'''low confidence sentence: Joe Biden attended the University of Pennsylvania, where he earned a law degree
passage 1: Joe Biden attended the University of Delaware, where he graduated in 1965 with a Bachelor of Arts in history and political science.
passage 2: After completing his undergraduate degree, Biden attended Syracuse University College of Law, where he earned a law degree in 1968.
passage 3: Joe Biden began his political career shortly after law school, becoming one of the youngest senators in U.S. history when he was elected to the Senate in 1972.
passage 4: Throughout his long political career, Biden served as the Vice President of the United States from 2009 to 2017 under President Barack Obama, and later became the 46th President of the United States in 2021.
passage 5: Biden's time at Syracuse University was marked by a plagiarism controversy, but he eventually graduated and went on to start his political career.
new sentence: He graduated from the University of Delaware in 1965 with a Bachelor of Arts in history and political science.

low confidence sentence: {y_t}
{evidences}
new sentence:'''
    return for_make_new_sent

def llm_based_clf_prompt(q):
    return_q = f'''Instruction: If you can answer the question, print 1. If you can't answer, print 0. Don't generate additional answers.
Question: Riverdance is part of a traditional dance from origination from which country?
Answer: 1

Question: Sir Harry Donald Secombe appeared notably as Mr Buble in a 1968 British musical drama film directed by who?
Answer: 0

Question: What did the population of Russians do between 1926-1959?
Answer: 0

Question: Which former mentor joined the judges' panel this season?
Answer: 1

Question: {q}
Answer:'''
    return return_q
