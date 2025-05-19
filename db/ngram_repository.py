from sqlalchemy.orm import Session
from db.ngram_index import NGramIndex as NGramIndexDB
import json
from ongoing_process_state.n_gram_index import NGramIndex

def save_n_gram_index_to_db(n_gram_index: NGramIndex, process_id: str, db: Session):
    for prefix, marking_ids in n_gram_index.markings.items():
        prefix_str = ",".join(prefix)
        marking_str = json.dumps(list(marking_ids))

        entry = NGramIndexDB(
            process_id=process_id,
            prefix=prefix_str,
            marking=marking_str,
        )
        db.merge(entry)
    db.commit()