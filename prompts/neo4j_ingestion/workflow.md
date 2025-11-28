# Neo4j Schema & Markdown Ingestion Playbook

This doc captures the exact steps we used to stand up the Neo4j schema and load every Markdown profile under `input/env/**/*.md` into the graph. Future agents can run the same workflow end-to-end without re‑deriving the process.

## 1. Prerequisites
- Ensure `.sercrets/secrets.toml` has a valid `[neo4j]` block (uri, username, password, database).
- Stay inside the repo root `/root/projects/TianGong-AI-Workspace` and rely on `uv` to execute Python (never `pip`/`poetry`).
- Neo4j instance must allow constraint/index creation; Chinese full‑text search requires the CJK analyzer feature.

## 2. Schema Bootstrapping & Data Loader
We execute a single Python script via `uv run python <<'PY' ... PY`. The script performs:
1. Reads Neo4j credentials from `.sercrets/secrets.toml`.
2. Walks `input/env/*.md`, extracting:
   - Core profile fields (name, title, contact, department, work location).
   - Biography and research sections (split into structured `ResearchArea` nodes and derived `Keyword` nodes).
   - Project bullet lines containing trigger terms (`项目/课题/计划/专项/基金`) with sponsor/role heuristics.
   - Publication blocks under the `## …学术成果` heading, preserving raw Markdown for provenance.
3. Builds stable `uid`s using `slugify(prefix, text)` ensuring idempotent `MERGE` operations.
4. Applies schema: uniqueness constraints for every label plus two full‑text indexes with the CJK analyzer.
5. Upserts nodes/relationships with provenance properties (`source_doc`, `doc_hash`, `extracted_at`), including:
   - `(:Professor)-[:BELONGS_TO]->(:Department)-[:PART_OF]->(:Institution)`
   - `(:Professor)-[:SPECIALIZES_IN]->(:ResearchArea)`
   - `(:Professor)-[:MENTIONS_KEYWORD]->(:Keyword)-[:ALIGNS_WITH]->(:ResearchArea)`
   - `(:Professor)-[:WORKS_ON]->(:Project)-[:ADDRESSES]->(:ResearchArea)`
   - `(:Publication)-[:AUTHORED_BY]->(:Professor)` and `(:Publication)-[:TAGGED_WITH]->(:ResearchArea)`

> **Command**
> ```bash
> uv run python - <<'PY'
> import datetime
> import hashlib
> import re
> import unicodedata
> from pathlib import Path
> from typing import Any, Dict, Iterable, List, Optional, Tuple
> 
> import tomllib
> from neo4j import GraphDatabase
> 
> PROJECT_ROOT = Path('.').resolve()
> DOC_ROOT = PROJECT_ROOT / 'input' / 'env'
> SECRETS_PATH = PROJECT_ROOT / '.sercrets' / 'secrets.toml'
> TIMESTAMP = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
> 
> FIELDS_WITH_COLON = [
>     ('姓名', 'name'),
>     ('职称', 'title'),
>     ('职务', 'position'),
>     ('邮箱', 'email'),
>     ('电话', 'phone'),
>     ('工作地点', 'work_location'),
>     ('二级单位', 'department'),
> ]
> 
> SECTION_LABELS = {
>     'bio': ['个人简介'],
>     'research': ['研究领域', '研究方向'],
> }
> 
> PROJECT_KEYWORDS = ('项目', '课题', '计划', '专项', '基金')
> SPONSOR_KEYWORDS = ('国家', '省', '市', '基金', '计划', '工程', '学院', '研究院')
> 
> PUBLICATION_HEADER_PATTERN = re.compile(r'^## .*学术成果.*$', re.MULTILINE)
> SECTION_LINE_PATTERN = re.compile(r'^\s*(\d+)\.(.+)$', re.MULTILINE)
> FIELD_PATTERN_CACHE: Dict[str, re.Pattern[str]] = {}
> 
> 
> def load_secrets() -> Dict[str, Any]:
>     if not SECRETS_PATH.exists():
>         raise SystemExit('Secrets file is missing: .sercrets/secrets.toml')
>     with SECRETS_PATH.open('rb') as fh:
>         data = tomllib.load(fh)
>     neo = data.get('neo4j')
>     if not neo:
>         raise SystemExit('Neo4j credentials not configured in secrets.')
>     return neo
> 
> 
> def slugify(prefix: str, text: str) -> str:
>     base = unicodedata.normalize('NFKD', text or '').encode('ascii', 'ignore').decode('ascii').lower()
>     base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
>     if base:
>         return f"{prefix}_{base}"
>     digest = hashlib.sha1((text or prefix).encode('utf-8')).hexdigest()[:12]
>     return f"{prefix}_{digest}"
> 
> 
> def clean_value(value: Optional[str]) -> Optional[str]:
>     if not value:
>         return None
>     stripped = value.strip()
>     if not stripped or stripped in {'无', '暂无'}:
>         return None
>     return stripped
> 
> 
> def extract_field(text: str, label: str) -> Optional[str]:
>     pattern = FIELD_PATTERN_CACHE.get(label)
>     if not pattern:
>         pattern = re.compile(rf'^\s*\d+\.\s*{re.escape(label)}\s*(?:[:：]\s*)?(.*)$', re.MULTILINE)
>         FIELD_PATTERN_CACHE[label] = pattern
>     match = pattern.search(text)
>     if not match:
>         return None
>     return match.group(1).strip()
> 
> 
> def extract_section(text: str, labels: Iterable[str]) -> Optional[str]:
>     for label in labels:
>         pattern = re.compile(rf'^\s*(\d+)\.\s*{re.escape(label)}\s*(?:[:：].*)?$', re.MULTILINE)
>         match = pattern.search(text)
>         if not match:
>             continue
>         start = match.end()
>         current_index = int(match.group(1))
>         end = len(text)
>         for sec_match in SECTION_LINE_PATTERN.finditer(text, start):
>             next_index = int(sec_match.group(1))
>             if next_index > current_index:
>                 end = sec_match.start()
>                 break
>         section_text = text[start:end].strip()
>         if section_text:
>             return section_text
>     return None
> 
> 
> def split_research_areas(text: Optional[str]) -> List[str]:
>     if not text:
>         return []
>     areas: List[str] = []
>     for line in text.splitlines():
>         stripped = line.strip().strip('•*- \t\u2022')
>         stripped = re.sub(r'^\d+(?:\\)?[\.).、］】]?', '', stripped).strip()
>         if not stripped:
>             continue
>         if stripped.endswith(':') or stripped.endswith('：'):
>             continue
>         areas.append(stripped)
>     if not areas:
>         block = text.strip()
>         if block:
>             areas = [blk.strip() for blk in re.split(r'[；;]', block) if blk.strip()]
>     return list(dict.fromkeys(areas))
> 
> 
> def derive_keywords(areas: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
>     keywords: Dict[str, Dict[str, Any]] = {}
>     for area_uid, area_text in areas:
>         raw_tokens = re.split(r'[、，,；;\s]', area_text)
>         for token in raw_tokens:
>             term = token.strip().strip('()（）"“”')
>             if len(term) < 2:
>                 continue
>             lang = 'zh' if re.search(r'[\u4e00-\u9fff]', term) else 'en'
>             kw_uid = slugify('kw', term)
>             keywords[kw_uid] = {
>                 'uid': kw_uid,
>                 'term': term,
>                 'lang': lang,
>                 'area_uid': area_uid,
>             }
>     return list(keywords.values())
> 
> 
> def normalize_projects(text: str) -> List[Dict[str, Any]]:
>     projects: Dict[str, Dict[str, Any]] = {}
>     if not text:
>         return []
>     for line in text.splitlines():
>         stripped = line.strip()
>         if not stripped:
>             continue
>         if not re.match(r'^\d+[\).、．]', stripped):
>             continue
>         normalized = re.sub(r'^\d+[\).、．]\s*', '', stripped)
>         if not any(key in normalized for key in PROJECT_KEYWORDS):
>             continue
>         parts = [part.strip() for part in re.split(r'[，,]', normalized) if part.strip()]
>         if not parts:
>             continue
>         title = parts[0]
>         sponsor = None
>         role = None
>         for part in parts[1:]:
>             if not sponsor and any(key in part for key in SPONSOR_KEYWORDS):
>                 sponsor = part
>             if '负责人' in part or '主持' in part:
>                 role = part
>         start_date, end_date = parse_date_range(normalized)
>         project_uid = slugify('proj', title)
>         projects[project_uid] = {
>             'uid': project_uid,
>             'title': title,
>             'sponsor': sponsor,
>             'role': role,
>             'start_date': start_date,
>             'end_date': end_date,
>             'source_line': normalized,
>         }
>     return list(projects.values())
> 
> 
> def parse_date_range(text: str) -> Tuple[Optional[str], Optional[str]]:
>     match = re.search(r'(20\d{2})(?:[.年-]?(\d{1,2}))?(?:[.月-]?(\d{1,2}))?[～~-](20\d{2})(?:[.年-]?(\d{1,2}))?(?:[.月-]?(\d{1,2}))?', text)
>     if not match:
>         return None, None
>     start = build_date(match.group(1), match.group(2), match.group(3))
>     end = build_date(match.group(4), match.group(5), match.group(6))
>     return start, end
> 
> 
> def build_date(year: Optional[str], month: Optional[str], day: Optional[str]) -> Optional[str]:
>     if not year:
>         return None
>     out = year
>     if month:
>         out += f'-{month.zfill(2)}'
>     if day:
>         out += f'-{day.zfill(2)}'
>     return out
> 
> 
> def parse_publications(text: str) -> List[Dict[str, Any]]:
>     if not text:
>         return []
>     header_match = PUBLICATION_HEADER_PATTERN.search(text)
>     content = text[header_match.end():].strip() if header_match else text.strip()
>     if not content:
>         return []
>     entries: List[List[str]] = []
>     current: List[str] = []
>     for line in content.splitlines():
>         if re.match(r'^\s*\d+(?:\\)?\.', line):
>             if current:
>                 entries.append(current)
>             current = [line]
>         else:
>             if not current and not line.strip():
>                 continue
>             current.append(line)
>     if current:
>         entries.append(current)
>     publications: List[Dict[str, Any]] = []
>     for block in entries:
>         if not block:
>             continue
>         first_line = block[0]
>         title = re.sub(r'^\s*\d+(?:\\)?\.\s*', '', first_line).strip()
>         authors: List[str] = []
>         venue_line = None
>         abstract_lines: List[str] = []
>         mode = 'authors'
>         for line in block[1:]:
>             stripped = line.strip()
>             if not stripped:
>                 continue
>             if stripped.lower().startswith('abstract'):
>                 mode = 'abstract'
>                 continue
>             if mode == 'authors':
>                 if venue_line is None and re.search(r'（20\d{2}）', stripped):
>                     venue_line = stripped
>                 else:
>                     cleaned = stripped.rstrip(',，')
>                     if cleaned:
>                         authors.append(cleaned)
>             else:
>                 abstract_lines.append(stripped)
>         venue = None
>         year = None
>         if venue_line:
>             venue_match = re.search(r'（(20\d{2})）', venue_line)
>             if venue_match:
>                 year = int(venue_match.group(1))
>                 venue = venue_line.split('（')[0].strip()
>             else:
>                 venue = venue_line.strip()
>         publication_uid = slugify('pub', title or block[0])
>         publications.append({
>             'uid': publication_uid,
>             'title': title,
>             'authors': authors,
>             'venue': venue,
>             'year': year,
>             'abstract': '\n'.join(abstract_lines).strip() if abstract_lines else None,
>             'raw': '\n'.join(block).strip(),
>         })
>     return publications
> 
> 
> def parse_markdown(path: Path) -> Dict[str, Any]:
>     text = path.read_text(encoding='utf-8')
>     second_header_idx = text.find('\n## ')
>     info_block = text if second_header_idx == -1 else text[:second_header_idx]
>     publications_block = '' if second_header_idx == -1 else text[second_header_idx:]
> 
>     data: Dict[str, Any] = {
>         'path': str(path),
>         'doc_hash': hashlib.sha1(text.encode('utf-8')).hexdigest(),
>     }
>     for label, key in FIELDS_WITH_COLON:
>         value = clean_value(extract_field(info_block, label))
>         data[key] = value
> 
>     bio_text = extract_section(info_block, SECTION_LABELS['bio'])
>     data['bio'] = bio_text.strip() if bio_text else None
> 
>     research_text = extract_section(info_block, SECTION_LABELS['research'])
>     areas = split_research_areas(research_text) if research_text else []
>     area_records = [(slugify('ra', entry), entry) for entry in areas]
>     data['research_areas'] = area_records
>     data['keywords'] = derive_keywords(area_records)
>     data['projects'] = normalize_projects(info_block)
>     data['publications'] = parse_publications(publications_block)
>     data['raw_research_text'] = research_text
>     return data
> 
> 
> def tenure_status_from_title(title: Optional[str]) -> Optional[str]:
>     if not title:
>         return None
>     if '助理' in title:
>         return 'assistant'
>     if '副' in title:
>         return 'associate'
>     if '教授' in title:
>         return 'professor'
>     if '研究员' in title:
>         return 'research'
>     return None
> 
> 
> def hash_email(email: Optional[str]) -> Optional[str]:
>     if not email or '@' not in email:
>         return None
>     return hashlib.sha256(email.lower().encode('utf-8')).hexdigest()
> 
> 
> def ingest_professor(session, record: Dict[str, Any]) -> None:
>     name = record.get('name')
>     if not name:
>         return
>     department_name = record.get('department') or record.get('work_location') or Path(record['path']).stem
>     institution_name = record.get('work_location') if record.get('work_location') and ('大学' in record['work_location'] or '学院' in record['work_location']) else '清华大学'
>     dept_level = 'research_institute' if '教研所' in department_name else ('center' if '中心' in department_name else 'department')
>     prof_uid = slugify('prof', f"{name}_{department_name}")
>     dept_uid = slugify('dept', department_name)
>     inst_uid = slugify('inst', institution_name)
>     email_hash = hash_email(record.get('email'))
>     titles = [item.strip() for item in re.split(r'[、，,;/]', record.get('title') or '') if item.strip()]
>     positions = [item.strip() for item in re.split(r'[、，,;/]', record.get('position') or '') if item.strip()]
> 
>     session.run(
>         """
>         MERGE (inst:Institution {uid: $inst_uid})
>           ON CREATE SET inst.created_at = $ts
>         SET inst.name_zh = $inst_name,
>             inst.updated_at = $ts
>         MERGE (dept:Department {uid: $dept_uid})
>           ON CREATE SET dept.created_at = $ts
>         SET dept.name_zh = $dept_name,
>             dept.level = $dept_level,
>             dept.parent_unit = $dept_parent,
>             dept.updated_at = $ts
>         MERGE (dept)-[:PART_OF]->(inst)
>         MERGE (prof:Professor {uid: $prof_uid})
>           ON CREATE SET prof.created_at = $ts
>         SET prof.name_zh = $prof_name,
>             prof.titles = $titles,
>             prof.positions = $positions,
>             prof.tenure_status = $tenure_status,
>             prof.email_hash = $email_hash,
>             prof.phone = $phone,
>             prof.work_location = $work_location,
>             prof.bio_text = $bio,
>             prof.doc_hash = $doc_hash,
>             prof.source_doc = $source_doc,
>             prof.extracted_at = $ts
>         MERGE (prof)-[:BELONGS_TO {source_doc: $source_doc, extracted_at: $ts}]->(dept)
>         """,
>         {
>             'inst_uid': inst_uid,
>             'inst_name': institution_name,
>             'dept_uid': dept_uid,
>             'dept_name': department_name,
>             'dept_level': dept_level,
>             'dept_parent': record.get('work_location'),
>             'prof_uid': prof_uid,
>             'prof_name': name,
>             'titles': titles or None,
>             'positions': positions or None,
>             'tenure_status': tenure_status_from_title(record.get('title')),
>             'email_hash': email_hash,
>             'phone': record.get('phone'),
>             'work_location': record.get('work_location'),
>             'bio': record.get('bio'),
>             'doc_hash': record['doc_hash'],
>             'source_doc': record['path'],
>             'ts': TIMESTAMP,
>         },
>     )
> 
>     area_uids: List[str] = []
>     for area_uid, area_name in record['research_areas']:
>         area_uids.append(area_uid)
>         session.run(
>             """
>             MATCH (prof:Professor {uid: $prof_uid})
>             MERGE (area:ResearchArea {uid: $area_uid})
>               ON CREATE SET area.created_at = $ts
>             SET area.name_zh = $area_name,
>                 area.description_zh = $area_name,
>                 area.updated_at = $ts
>             MERGE (prof)-[rel:SPECIALIZES_IN]->(area)
>             SET rel.source_doc = $source_doc,
>                 rel.extracted_at = $ts,
>                 rel.confidence = 0.85
>             """,
>             {
>                 'prof_uid': prof_uid,
>                 'area_uid': area_uid,
>                 'area_name': area_name,
>                 'source_doc': record['path'],
>                 'ts': TIMESTAMP,
>             },
>         )
> 
>     for keyword in record['keywords']:
>         session.run(
>             """
>             MATCH (prof:Professor {uid: $prof_uid})
>             MERGE (kw:Keyword {uid: $kw_uid})
>               ON CREATE SET kw.created_at = $ts
>             SET kw.term = $term,
>                 kw.lang = $lang,
>                 kw.updated_at = $ts
>             MERGE (prof)-[rel:MENTIONS_KEYWORD]->(kw)
>             SET rel.weight = 1.0,
>                 rel.source_doc = $source_doc,
>                 rel.extracted_at = $ts
>             WITH kw
>             MATCH (area:ResearchArea {uid: $area_uid})
>             MERGE (kw)-[:ALIGNS_WITH {source_doc: $source_doc, extracted_at: $ts}]->(area)
>             """,
>             {
>                 'prof_uid': prof_uid,
>                 'kw_uid': keyword['uid'],
>                 'term': keyword['term'],
>                 'lang': keyword['lang'],
>                 'area_uid': keyword['area_uid'],
>                 'source_doc': record['path'],
>                 'ts': TIMESTAMP,
>             },
>         )
> 
>     for project in record['projects']:
>         session.run(
>             """
>             MATCH (prof:Professor {uid: $prof_uid})
>             MERGE (project:Project {uid: $project_uid})
>               ON CREATE SET project.created_at = $ts
>             SET project.title = $title,
>                 project.sponsor = $sponsor,
>                 project.start_date = $start_date,
>                 project.end_date = $end_date,
>                 project.updated_at = $ts,
>                 project.source_line = $source_line
>             MERGE (prof)-[rel:WORKS_ON]->(project)
>             SET rel.role = $role,
>                 rel.source_doc = $source_doc,
>                 rel.extracted_at = $ts
>             WITH project
>             UNWIND $area_uids AS area_uid
>             MATCH (area:ResearchArea {uid: area_uid})
>             MERGE (project)-[:ADDRESSES {source_doc: $source_doc, extracted_at: $ts}]->(area)
>             """,
>             {
>                 'prof_uid': prof_uid,
>                 'project_uid': project['uid'],
>                 'title': project['title'],
>                 'sponsor': project.get('sponsor'),
>                 'start_date': project.get('start_date'),
>                 'end_date': project.get('end_date'),
>                 'source_line': project.get('source_line'),
>                 'role': project.get('role'),
>                 'source_doc': record['path'],
>                 'area_uids': area_uids or [],
>                 'ts': TIMESTAMP,
>             },
>         )
> 
>     for publication in record['publications']:
>         session.run(
>            """
>            MATCH (prof:Professor {uid: $prof_uid})
>            MERGE (pub:Publication {uid: $pub_uid})
>              ON CREATE SET pub.created_at = $ts
>            SET pub.title = $title,
>                pub.venue = $venue,
>                pub.year = $year,
>                pub.abstract = $abstract,
>                pub.raw_entry = $raw,
>                pub.updated_at = $ts
>            MERGE (pub)-[:AUTHORED_BY {source_doc: $source_doc, extracted_at: $ts}]->(prof)
>            WITH pub
>            UNWIND $area_uids AS area_uid
>            MATCH (area:ResearchArea {uid: area_uid})
>            MERGE (pub)-[:TAGGED_WITH {source_doc: $source_doc, extracted_at: $ts}]->(area)
>            """,
>            {
>                'prof_uid': prof_uid,
>                'pub_uid': publication['uid'],
>                'title': publication.get('title'),
>                'venue': publication.get('venue'),
>                'year': publication.get('year'),
>                'abstract': publication.get('abstract'),
>                'raw': publication.get('raw'),
>                'source_doc': record['path'],
>                'area_uids': area_uids or [],
>                'ts': TIMESTAMP,
>            },
>        )
> 
> 
> def apply_schema(session) -> None:
>     statements = [
>         "CREATE CONSTRAINT professor_uid IF NOT EXISTS FOR (p:Professor) REQUIRE p.uid IS UNIQUE",
>         "CREATE CONSTRAINT department_uid IF NOT EXISTS FOR (d:Department) REQUIRE d.uid IS UNIQUE",
>         "CREATE CONSTRAINT institution_uid IF NOT EXISTS FOR (i:Institution) REQUIRE i.uid IS UNIQUE",
>         "CREATE CONSTRAINT research_area_uid IF NOT EXISTS FOR (r:ResearchArea) REQUIRE r.uid IS UNIQUE",
>         "CREATE CONSTRAINT keyword_uid IF NOT EXISTS FOR (k:Keyword) REQUIRE k.uid IS UNIQUE",
>         "CREATE CONSTRAINT project_uid IF NOT EXISTS FOR (p:Project) REQUIRE p.uid IS UNIQUE",
>         "CREATE CONSTRAINT publication_uid IF NOT EXISTS FOR (p:Publication) REQUIRE p.uid IS UNIQUE",
>         """
>         CREATE FULLTEXT INDEX professorChineseBio IF NOT EXISTS
>         FOR (p:Professor)
>         ON EACH [p.bio_text]
>         OPTIONS { indexConfig: { `fulltext.analyzer`: 'cjk' } }
>         """,
>         """
>         CREATE FULLTEXT INDEX researchAreaChinese IF NOT EXISTS
>         FOR (r:ResearchArea)
>         ON EACH [r.description_zh]
>         OPTIONS { indexConfig: { `fulltext.analyzer`: 'cjk' } }
>         """,
>     ]
>     for stmt in statements:
>         session.run(stmt)
> 
> 
> def main() -> None:
>     neo = load_secrets()
>     driver = GraphDatabase.driver(neo['uri'], auth=(neo['username'], neo['password']))
>     database = neo.get('database') or 'neo4j'
>     documents = sorted(DOC_ROOT.glob('*.md'))
>     if not documents:
>         raise SystemExit('No Markdown files found under input/env.')
>     total_ingested = 0
>     with driver.session(database=database) as session:
>         apply_schema(session)
>         for doc in documents:
>             record = parse_markdown(doc)
>             if not record.get('name'):
>                 print(f"Skipping {doc.name}: missing name field.")
>                 continue
>             ingest_professor(session, record)
>             total_ingested += 1
>             print(f"Ingested {record['name']} from {doc.name}.")
>     driver.close()
>     print(f"Completed ingestion for {total_ingested} profiles.")
> 
> 
> if __name__ == '__main__':
>     main()
> PY
> ```

## 3. Verification Queries
Immediately after ingestion, validate counts and intersection logic:

```bash
uv run python - <<'PY'
import tomllib
from pathlib import Path
from neo4j import GraphDatabase

cfg = tomllib.load(open('.sercrets/secrets.toml','rb'))['neo4j']
driver = GraphDatabase.driver(cfg['uri'], auth=(cfg['username'], cfg['password']))
with driver.session(database=cfg.get('database') or 'neo4j') as session:
    metrics = {
        'Professor': session.run('MATCH (p:Professor) RETURN count(p)').single().value(),
        'Department': session.run('MATCH (d:Department) RETURN count(d)').single().value(),
        'ResearchArea': session.run('MATCH (r:ResearchArea) RETURN count(r)').single().value(),
        'Keyword': session.run('MATCH (k:Keyword) RETURN count(k)').single().value(),
        'Project': session.run('MATCH (p:Project) RETURN count(p)').single().value(),
        'Publication': session.run('MATCH (p:Publication) RETURN count(p)').single().value(),
    }
    print('Node counts', metrics)
    rels = {
        'SPECIALIZES_IN': session.run('MATCH ()-[r:SPECIALIZES_IN]->() RETURN count(r)').single().value(),
        'MENTIONS_KEYWORD': session.run('MATCH ()-[r:MENTIONS_KEYWORD]->() RETURN count(r)').single().value(),
        'WORKS_ON': session.run('MATCH ()-[r:WORKS_ON]->() RETURN count(r)').single().value(),
        'AUTHORED_BY': session.run('MATCH ()-[r:AUTHORED_BY]->() RETURN count(r)').single().value(),
    }
    print('Relationship counts', rels)
    rows = session.run('''
        MATCH (a:ResearchArea)<-[:SPECIALIZES_IN]-(p:Professor)-[:SPECIALIZES_IN]->(b:ResearchArea)
        WHERE a.uid < b.uid
        RETURN a.name_zh AS area_a, b.name_zh AS area_b, count(DISTINCT p) AS professors
        ORDER BY professors DESC LIMIT 10
    ''').data()
    print('Cross-discipline pairs', rows)
driver.close()
PY
```

Expected (as of 2025‑02‑20): 14 `Professor` nodes covering every Markdown file, 54 `ResearchArea` nodes, 218 keywords, 1 project, 1 749 publications, and populated cross-area pairs.

## 4. Known Follow-ups
1. Broaden project extraction so more than the first numeric bullet is captured.
2. Filter research-area noise (one entry captured招聘文本) before creating nodes.
3. Optionally derive `COOPERATES_WITH` edges from shared projects/publications for downstream analytics.

Stick to this playbook whenever we re-hydrate the knowledge graph so results stay deterministic and reproducible.
