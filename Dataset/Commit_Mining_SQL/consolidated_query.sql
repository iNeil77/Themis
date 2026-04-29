-- =============================================================================
-- Consolidated Commit Mining Query
-- =============================================================================
-- Extracts single-file code commits from bigquery-public-data.github_repos.
--
-- Joins:  languages → licenses → commits (unnested repo_name) → difference
--
-- Filters:
--   - 12 permissive open-source licenses (MIT, Apache-2.0, BSD, etc.)
--   - 9 programming languages (Python, Java, JavaScript, C, C#, C++, TS, Go, Ruby)
--   - Non-trivial commit messages (10 < length < 15000 chars)
--   - Blocklist of ~50 low-signal messages ("initial commit", "wip", etc.)
--   - Pattern exclusions for merge commits and CI pushes
--   - Same-path constraint (old_path = new_path): file modified in place
--   - Single-file constraint (HAVING COUNT(DISTINCT old_path) = 1)
--
-- Output columns:
--   commit    – git SHA
--   subject   – first line of commit message
--   message   – full commit message body
--   repos     – comma-separated list of repos containing this commit
--   license   – SPDX license identifier
--   old_file  – path of changed file (pre-commit)
--   new_file  – path of changed file (post-commit)
--   unix_time – committer timestamp (seconds since epoch)
-- =============================================================================

SELECT c.commit,
    c.subject,
    c.message,
    STRING_AGG(DISTINCT unnested_repo_name) AS repos,
    l.license,
    d.old_path AS old_file,
    d.new_path AS new_file,
    c.committer.time_sec AS unix_time
FROM `bigquery-public-data.github_repos.languages` AS lang_table,
    UNNEST(language) AS lang
    JOIN `bigquery-public-data.github_repos.licenses` AS l ON l.repo_name = lang_table.repo_name
    JOIN (
        SELECT *,
            unnested_repo_name
        FROM `bigquery-public-data.github_repos.commits`,
            UNNEST(repo_name) AS unnested_repo_name
    ) c ON c.unnested_repo_name = lang_table.repo_name,
    UNNEST(c.difference) AS d
WHERE l.license IN (
        'mit',
        'artistic-2.0',
        'isc',
        'cc0-1.0',
        'epl-1.0',
        'mpl-2.0',
        'unlicense',
        'apache-2.0',
        'bsd-3-clause',
        'agpl-3.0',
        'lgpl-2.1',
        'bsd-2-clause'
    )
    AND lang.name IN (
        'Python',
        'Java',
        'JavaScript',
        'C',
        'C#',
        'C++',
        'TypeScript',
        'Go',
        'Ruby',
    )
    AND LENGTH(c.message) > 10
    AND LENGTH(c.message) < 15000
    AND LOWER(c.message) NOT IN (
        'update readme.md',
        'initial commit',
        'update',
        'mirroring from micro.blog.',
        'update data.json',
        'update data.js',
        'add files via upload',
        'update readme',
        "can't you see i'm updating the time?",
        'dummy',
        'update index.html',
        'first commit',
        'create readme.md',
        'heartbeat update',
        'updated readme',
        'update log',
        'test',
        'no message',
        'readme',
        'wip',
        'updates',
        'commit',
        'update _config.yaml',
        'testing',
        'tweak',
        'tweaks',
        'modified',
        'edited',
        'yolo commit',
        'yolo',
        'made it work',
        'work in progress',
        'fixing',
        'for review',
        'my changes',
        'revised',
        'addressed comments',
        'placeholder',
        'test commit',
        'trying something',
        'experimental changes',
        'hack',
        'do not merge',
        'various updates',
        'stuff'
    )
    AND LOWER(c.message) NOT LIKE '%pi push%'
    AND LOWER(c.message) NOT LIKE '%push pi%'
    AND LOWER(c.message) NOT LIKE 'merge%'
    AND d.old_path = d.new_path
    AND d.old_path IS NOT NULL
    AND d.new_path IS NOT NULL
GROUP BY c.commit,
    c.subject,
    c.message,
    l.license,
    d.old_path,
    d.new_path,
    c.committer.time_sec
HAVING COUNT(DISTINCT d.old_path) = 1
