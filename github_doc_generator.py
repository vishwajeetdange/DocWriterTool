import datetime
import fnmatch
import io
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import markdown
import zipfile
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
import openai
import concurrent.futures
from typing import Dict, List, Set, Any
import re
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from concurrent.futures import ThreadPoolExecutor, as_completed
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from collections import defaultdict
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class DocumentationConfig:
    """
    Configuration class for documentation generation options.
    """
    def __init__(
        self,
        include_overview: bool = True,
        include_executive_summary: bool = True,
        include_tech_stack: bool = True,
        include_code_structure: bool = False,
        include_loc_analysis: bool = True,
        include_complexity_analysis: bool = True,
        include_features: bool = True,
        include_dependencies: bool = True,
        include_issues: bool = True,
        include_sql_objects: bool = True,
        include_class_diagram: bool = True,
        include_flow_diagram: bool = True,
        include_er_diagram: bool = True,
        include_reference_architecture: bool = False,
        include_loc_chart: bool = True,
        include_complexity_charts: bool = True,
        include_line_by_line_docs: bool = True,  # New option for line-by-line documentation
        max_files_to_analyze: int = 5,
        diagram_depth: int = 2  # 1=basic, 2=detailed, 3=comprehensive
    ):
        self.include_overview = include_overview
        self.include_executive_summary = include_executive_summary
        self.include_tech_stack = include_tech_stack
        self.include_code_structure = include_code_structure
        self.include_loc_analysis = include_loc_analysis
        self.include_complexity_analysis = include_complexity_analysis
        self.include_features = include_features
        self.include_dependencies = include_dependencies
        self.include_issues = include_issues
        self.include_sql_objects = include_sql_objects
        self.include_class_diagram = include_class_diagram
        self.include_flow_diagram = include_flow_diagram
        self.include_er_diagram = include_er_diagram
        self.include_reference_architecture = include_reference_architecture
        self.include_loc_chart = include_loc_chart
        self.include_complexity_charts = include_complexity_charts
        self.include_line_by_line_docs = include_line_by_line_docs
        self.max_files_to_analyze = max_files_to_analyze
        self.diagram_depth = diagram_depth


class GitHubDocGenerator:
    def __init__(
        self,
        deployment_name: str,
        api_key: str,
        api_base: str,
        github_token: str = None,
        output_dir: str = None,
        config: DocumentationConfig = None
    ):
        self.contents = {}  # Add shared contents dictionary
        self.file_analysis_cache = {}
        self.max_file_size = 100000  # 100KB

        # Azure OpenAI setup for GPT-4-turbo
        self.deployment_name = deployment_name
        openai.api_type = "azure"
        openai.api_version = ""
        openai.api_key = api_key
        openai.api_base = api_base

        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output_docs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up cache directory
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Set configuration
        self.config = config if config is not None else DocumentationConfig()

        self.exclude_patterns = {
            "*.min.js",
            "*.min.css",
            "node_modules/*",
            "dist/*",
            "build/*",
            "*.pyc",
            "__pycache__/*",
            "*.log",
            "*.map",
            "vendor/*",
            ".git/*",
            "venv/*",
            "env/*",
            ".vscode/*",
            ".idea/*",
            "*.mo",
            "*.po",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.svg",
            "*.ico",
            "*.woff*",
            "*.ttf",
            "*.eot",
            "*.otf",
            "*.pdf",
            "*.zip",
            "*.tar.gz",
        }

        # Define supported file extensions
        self.supported_extensions = {
            # COBOL
            "cbl": "COBOL",
            "cob": "COBOL",
            "cpy": "COBOL Copybook",
            # C
            "c": "C",
            "h": "C Header",
            # C++
            "cpp": "C++",
            "cxx": "C++",
            "cc": "C++",
            "hpp": "C++ Header",
            # Java
            "java": "Java",
            # Frontend Technologies
            "html": "HTML",
            "htm": "HTML",
            "css": "CSS",
            "scss": "SASS",
            "sass": "SASS",
            "less": "LESS",
            "js": "JavaScript",
            "mjs": "JavaScript Module",
            "jsx": "React JavaScript",
            "ts": "TypeScript",
            "tsx": "TypeScript React",
            "vue": "Vue.js",
            "svelte": "Svelte",
            # Configuration files
            "yaml": "YAML",
            "yml": "YAML",
            "json": "JSON",
            "xml": "XML",
            "txt": "Text",
            # .NET Framework
            "cs": "C#",
            "vb": "Visual Basic .NET",
            "fs": "F#",
            "csproj": ".NET Project",
            "vbproj": "VB.NET Project",
            "fsproj": "F# Project",
            "config": ".NET Configuration",
            "dll": ".NET Assembly",
            "exe": ".NET Executable",
            "asax": "ASP.NET Handler",
            "aspx": "ASP.NET Web Form",
            "ascx": "ASP.NET User Control",
            "asmx": "ASP.NET Web Service",
            "ashx": "ASP.NET HTTP Handler",
            "master": "ASP.NET Master Page",
            "cshtml": "ASP.NET Razor",
            "vbhtml": "ASP.NET Razor (VB)",
            # Angular
            "component.ts": "Angular Component",
            "service.ts": "Angular Service",
            "module.ts": "Angular Module",
            "pipe.ts": "Angular Pipe",
            "directive.ts": "Angular Directive",
            "guard.ts": "Angular Route Guard",
            "resolver.ts": "Angular Resolver",
            "interceptor.ts": "Angular Interceptor",
            "animations.ts": "Angular Animations",
            "spec.ts": "Angular Test",
            # Blazor
            "razor": "Blazor Component",
            "blazor": "Blazor Application",
            "blazorlib": "Blazor Class Library",
            # Python
            "py": "Python",
            "pyw": "Python Windows Script",
            "pyc": "Python Compiled Bytecode",
            "pyo": "Python Optimized Bytecode",
            "pyd": "Python Extension Module",
            "pyi": "Python Type Hints",
            "pyx": "Cython",
            "pxd": "Cython Definition",
            "pxi": "Cython Include",
            # Django
            "djhtml": "Django HTML Template",
            "djt": "Django Template",
            "dtl": "Django Template Language",
            # Flask
            "flaskenv": "Flask Environment Variables",
            "wsgi": "WSGI Script",
            # SQL
            "sql": "SQL",
            "ddl": "SQL Data Definition Language",
            "dml": "SQL Data Manipulation Language",
            "pks": "SQL Package Specification",
            "pkb": "SQL Package Body",
            "trg": "SQL Trigger",
            "prc": "SQL Stored Procedure",
            "fnc": "SQL Function",
            "vw": "SQL View",
            "idx": "SQL Index",
            "seq": "SQL Sequence",
            # ABAP
            "abap": "ABAP",
            "abapgit": "ABAP Git",
            # AS/400 (IBM i)
            "rpg": "RPG",
            "rpgle": "RPG ILE",
            "sqlrpgle": "SQL RPG",
            "clp": "CL Program",
            "clle": "CL ILE",
            "dspf": "Display File",
            "prtf": "Printer File",
            "lf": "Logical File",
            "pf": "Physical File",
            "dds": "DDS Source",
            "bnd": "Binder Source",
            "copy": "Copy Member",
        }

        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Set up GitHub API access
        self.github_token = github_token
        self.is_authenticated = bool(github_token)

        # Base headers for GitHub API
        self.session.headers.update(
            {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "GitHub-Documentation-Generator",
            }
        )

        # Add authentication token if provided
        if self.github_token:
            self.session.headers.update({"Authorization": f"token {self.github_token}"})

    def _get_cache_key(self, content: str, analysis_type: str) -> str:
        """Generate a consistent cache key for analysis results."""
        import hashlib
        return hashlib.md5((content[:1000] + analysis_type).encode()).hexdigest()

    def _get_cached_analysis(self, cache_key: str) -> Optional[str]:
        """Retrieve cached analysis if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _cache_analysis(self, cache_key: str, analysis: str):
        """Cache analysis results."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f)
        except IOError as e:
            logging.error(f"Failed to cache analysis: {e}")

    def analyze_files_parallel(self, files_to_analyze: List[Tuple[str, dict]]) -> Dict[str, str]:
        """Analyze multiple files in parallel."""
        file_analyses = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit to 5 concurrent API calls
            futures = {
                executor.submit(
                    self.analyze_code_file, 
                    info['content'], 
                    info['language'], 
                    path
                ): path 
                for path, info in files_to_analyze
            }
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    file_analyses[path] = future.result()
                except Exception as e:
                    logging.error(f"Error analyzing {path}: {e}")
                    file_analyses[path] = f"Error analyzing file: {str(e)}"
        
        return file_analyses
    
    def select_key_files(self, contents: Dict[str, dict], max_files: int = 5) -> List[str]:
        """Select the most important files for analysis based on naming patterns and language types."""
        # Enhanced priority patterns for different languages
        priority_patterns = {
            "high": [
                r"main\.", r"app\.", r"application\.", r"startup\.", r"entry\.",
                r"driver\.", r"control\.", r"pgm[0-9]*\.", r"program\."
            ],
            "medium": [
                r"core\.", r"service\.", r"controller\.", r"manager\.",
                r"model\.", r"repository\.", r"api\.", r"handler\.",
                r"proc[0-9]*\.", r"function\.", r"module\."
            ],
            "low": [
                r"index\.", r"config\.", r"setup\.", r"init\.", r"util\.",
                r"helper\.", r"common\.", r"shared\.", r"base\."
            ]
        }
        
        # Language-specific priority patterns
        language_priorities = {
            "COBOL": [r"main", r"driver", r"control", r"file[0-9]*", r"report[0-9]*"],
            "RPG": [r"main", r"driver", r"control", r"pgm[0-9]*", r"proc[0-9]*"],
            "RPG ILE": [r"main", r"driver", r"control", r"pgm[0-9]*", r"proc[0-9]*"],
            "SQL RPG": [r"main", r"driver", r"control", r"pgm[0-9]*", r"proc[0-9]*"],
            "CL Program": [r"main", r"startup", r"control", r"job[0-9]*"],
            "CL ILE": [r"main", r"startup", r"control", r"job[0-9]*"],
            "DDS Source": [r".*dspf", r".*prtf", r".*lf", r".*pf", r"display", r"file"],
            "Display File": [r".*dspf", r"display", r"screen", r"panel"],
            "Printer File": [r".*prtf", r"print", r"report", r"output"],
            "Logical File": [r".*lf", r"logical", r"view", r"index"],
            "Physical File": [r".*pf", r"physical", r"table", r"master"],
            "Java": [r"main", r"application", r"controller", r"service", r"repository"],
            "C#": [r"main", r"program", r"startup", r"controller", r"service"],
            "Python": [r"main", r"app", r"__init__", r"views", r"models"],
            "SQL": [r"create", r"proc", r"function", r"trigger", r"view"]
        }
        
        high_priority = []
        medium_priority = []
        low_priority = []
        other_files = []
        
        for path, info in contents.items():
            if info["type"] != "file" or "language" not in info:
                continue
                
            filename = os.path.basename(path).lower()
            language = info["language"]
            
            # Check language-specific patterns first
            if language in language_priorities:
                lang_patterns = language_priorities[language]
                if any(re.search(pattern, filename, re.IGNORECASE) for pattern in lang_patterns):
                    high_priority.append((path, info))
                    continue
            
            # Check general priority patterns
            if any(re.search(pattern, filename) for pattern in priority_patterns["high"]):
                high_priority.append((path, info))
            elif any(re.search(pattern, filename) for pattern in priority_patterns["medium"]):
                medium_priority.append((path, info))
            elif any(re.search(pattern, filename) for pattern in priority_patterns["low"]):
                low_priority.append((path, info))
            else:
                other_files.append((path, info))
        
        # Sort by file size (larger files often more important) and take the best from each category
        high_priority.sort(key=lambda x: x[1].get("size", 0), reverse=True)
        medium_priority.sort(key=lambda x: x[1].get("size", 0), reverse=True)
        low_priority.sort(key=lambda x: x[1].get("size", 0), reverse=True)
        other_files.sort(key=lambda x: x[1].get("size", 0), reverse=True)
        
        # Select files with priority distribution
        selected_files = []
        
        # Take at least 60% from high priority
        high_count = min(len(high_priority), max(1, int(max_files * 0.6)))
        selected_files.extend(high_priority[:high_count])
        
        # Fill remaining slots with medium and low priority
        remaining_slots = max_files - len(selected_files)
        if remaining_slots > 0:
            medium_count = min(len(medium_priority), max(1, remaining_slots // 2))
            selected_files.extend(medium_priority[:medium_count])
            
            remaining_slots = max_files - len(selected_files)
            if remaining_slots > 0:
                selected_files.extend(low_priority[:remaining_slots])
        
        # If still not enough files, add from other_files
        remaining_slots = max_files - len(selected_files)
        if remaining_slots > 0:
            selected_files.extend(other_files[:remaining_slots])
        
        return selected_files

    def check_rate_limit(self) -> Tuple[bool, int, datetime.datetime]:
        try:
            response = self.session.get("https://api.github.com/rate_limit")
            if response.status_code == 200:
                rate_data = response.json()["resources"]["core"]
                remaining = rate_data["remaining"]
                reset_time = datetime.datetime.fromtimestamp(rate_data["reset"])

                # Log rate limit status
                limit_type = (
                    "Authenticated" if self.is_authenticated else "Unauthenticated"
                )
                logging.info(
                    f"{limit_type} rate limit status: {remaining} requests remaining, "
                    f"resets at {reset_time}"
                )

                return True, remaining, reset_time
            return False, 0, datetime.datetime.now()
        except Exception as e:
            logging.error(f"Error checking rate limit: {e}")
            return False, 0, datetime.datetime.now()

    def handle_rate_limit(self) -> bool:
        success, remaining, reset_time = self.check_rate_limit()

        if not success:
            return False

        if remaining == 0:
            wait_time = (reset_time - datetime.datetime.now()).total_seconds()
            if wait_time > 0:
                if self.is_authenticated:
                    # Authenticated users get higher rate limits, worth waiting
                    logging.warning(
                        f"Rate limit exceeded. Waiting {wait_time:.0f} seconds..."
                    )
                    time.sleep(min(wait_time + 1, 3600))  # Wait up to 1 hour
                    return True
                else:
                    # For unauthenticated users, suggest using authentication
                    logging.warning(
                        "Rate limit exceeded for unauthenticated access. "
                        "Consider using a GitHub token for higher limits."
                    )
                    return False
        return True

    def validate_github_url(self, github_link: str) -> Optional[Tuple[str, str]]:
        patterns = [
            r"https?://github\.com/([^/]+)/([^/]+)/?",
            r"https?://www\.github\.com/([^/]+)/([^/]+)/?",
            r"([^/]+)/([^/]+)",
        ]

        for pattern in patterns:
            match = re.match(pattern, github_link)
            if match:
                return match.group(1), match.group(2)
        return None

    def fetch_repository_contents(
        self, username: str, repo_name: str
    ) -> Dict[str, dict]:
        """
        Fetch repository contents, excluding large or unsupported files.
        """
        try:
            base_url = f"https://api.github.com/repos/{username}/{repo_name}/contents"
            contents = {}

            def fetch_directory_contents(
                url: str, path: str = "", retries: int = 3
            ) -> Dict[str, dict]:
                while retries > 0:
                    try:
                        response = self.session.get(url, timeout=30)
                        response.raise_for_status()
                        items = response.json()

                        for item in items:
                            item_path = os.path.join(path, item["name"])

                            if item["type"] == "file":
                                # Skip large files
                                if item["size"] > self.max_file_size:
                                    logging.warning(f"Skipping large file: {item_path}")
                                    continue

                                ext = item["name"].split(".")[-1].lower()
                                if ext in self.supported_extensions:
                                    try:
                                        file_response = self.session.get(
                                            item["download_url"], timeout=30
                                        )
                                        file_response.raise_for_status()
                                        contents[item_path] = {
                                            "content": file_response.text,
                                            "language": self.supported_extensions[ext],
                                            "size": item["size"],
                                            "type": "file",
                                        }
                                    except requests.exceptions.RequestException as e:
                                        logging.warning(
                                            f"Error fetching file {item_path}: {e}"
                                        )

                            elif item["type"] == "dir":
                                contents[item_path] = {"type": "directory", "items": []}
                                fetch_directory_contents(item["url"], item_path)

                        break  # Success, exit retry loop
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Error fetching contents for {url}: {e}")
                        retries -= 1
                        if retries > 0:
                            time.sleep(2 ** (3 - retries))  # Exponential backoff
                        else:
                            break

            # Start fetching from root
            fetch_directory_contents(base_url)
            return contents

        except Exception as e:
            logging.error(f"Error accessing repository: {e}")
            raise

    def analyze_code_file(self, content: str, language: str, file_path: str, max_length: int = 8000) -> str:
        try:
            cache_key = self._get_cache_key(content, f"analysis_{language}_{file_path}")
            cached_result = self._get_cached_analysis(cache_key)
            if cached_result is not None:
                logging.info(f"Using cached analysis for {file_path}")
                return cached_result

            if len(content) > max_length:
                sample_size = max_length // 3
                samples = [
                    content[:sample_size],
                    content[len(content)//2 - sample_size//2 : len(content)//2 + sample_size//2],
                    content[-sample_size:]
                ]
                sampled_content = "\n\n... [middle content omitted] ...\n\n".join(samples)
                logging.info(f"Sampling large file {file_path} ({len(content)} -> {len(sampled_content)} chars)")
            else:
                sampled_content = content

            analysis = self._analyze_with_language_config(sampled_content, language, file_path)
            self._cache_analysis(cache_key, analysis)
            return analysis

        except Exception as e:
            logging.error(f"Unexpected error analyzing {file_path}: {str(e)}")
            return f"Error analyzing file: {str(e)}"
        
    def batch_analyze_files(self, files: List[Tuple[str, dict]]) -> Dict[str, str]:
        """Batch analyze similar files together to reduce API calls."""
        # Group files by language
        files_by_language = defaultdict(list)
        for path, info in files:
            files_by_language[info["language"]].append((path, info))
        
        results = {}
        
        for lang, lang_files in files_by_language.items():
            if len(lang_files) == 1:
                # Single file - use regular analysis
                path, info = lang_files[0]
                results[path] = self.analyze_code_file(info["content"], lang, path)
            else:
                # Multiple files in same language - batch analyze
                prompt = f"""
                Analyze the following {lang} code files and provide concise documentation for each:
                
                {[f['path'] for f in lang_files]}
                
                For each file, provide:
                1. Brief purpose (1 sentence)
                2. Key components/functions
                3. Notable patterns
                4. Quality assessment (good/fair/poor)
                
                Format as a JSON dictionary with filenames as keys.
                """
                
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a code documentation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.3
                )
                
                try:
                    batch_results = json.loads(response["choices"][0]["message"]["content"])
                    results.update(batch_results)
                except:
                    # Fallback to individual analysis if batch fails
                    for path, info in lang_files:
                        results[path] = self.analyze_code_file(info["content"], lang, path)
        
        return results
    
    def generate_diagrams_parallel(self, contents: Dict[str, dict], repo_name: str) -> Dict[str, str]:
        """Generate all diagrams in parallel."""
        diagrams = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all diagram generation tasks
            future_class = executor.submit(self.generate_class_diagram, contents, repo_name)
            future_flow = executor.submit(self._generate_flow_diagram, contents, repo_name)
            future_er = executor.submit(self._generate_er_diagram, contents, repo_name)
            future_arch = executor.submit(self.generate_reference_architecture_diagram, contents, repo_name)
            
            # Collect results as they complete
            diagrams["class_diagram"] = future_class.result()
            diagrams["flow_diagram"] = future_flow.result()
            diagrams["er_diagram"] = future_er.result()
            diagrams["reference_architecture"] = future_arch.result()
        
        return diagrams

    def analyze_code_quality(self, contents: Dict[str, dict]) -> Dict:
        """
        Analyze overall code quality metrics and best practices.

        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.

        Returns:
            Dict: Analysis results with code quality metrics.
        """
        # Count files per language
        language_files = {}
        language_loc = {}
        total_loc = 0

        # Extract file metrics
        for path, info in contents.items():
            if info["type"] == "file" and "language" in info:
                lang = info["language"]
                loc = len(info["content"].split("\n"))

                if lang not in language_files:
                    language_files[lang] = 0
                    language_loc[lang] = 0

                language_files[lang] += 1
                language_loc[lang] += loc
                total_loc += loc

        # Prepare sample files for deeper analysis
        sample_files = []
        for path, info in contents.items():
            if info["type"] == "file" and "language" in info:
                # Take a few representative files of each main language
                lang = info["language"]
                if len(sample_files) < 10 and path not in sample_files:
                    sample_files.append(path)

        # Analyze code quality metrics with GPT-4
        code_samples = []
        for path in sample_files[:5]:  # Limit to 5 for API efficiency
            if path in contents and contents[path]["type"] == "file":
                code_samples.append(
                    {
                        "path": path,
                        "language": contents[path]["language"],
                        "snippet": contents[path]["content"][:1000],  # First 1000 chars
                    }
                )

        # Generate code quality report
        quality_prompt = f"""
        Analyze the code quality of this repository based on these metrics and sample files:
        
        Repository Statistics:
        - Total lines of code: {total_loc}
        - Files by language: {language_files}
        - Lines by language: {language_loc}
        
        Sample code snippets:
        {code_samples}
        
        Provide a comprehensive code quality assessment including:
        
        1. Code Quality Score (0-100)
        2. Top 5 Best Practices observed in the code
        3. Top 5 Code Issues/Anti-patterns identified
        4. Recommendations for improvement (specific and actionable)
        5. Technical Debt assessment
        
        Format the response as a structured Markdown report with clear sections.
        """

        response = openai.ChatCompletion.create(
            engine=self.deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior software architect with extensive expertise in static code analysis and quality assessment.",
                },
                {"role": "user", "content": quality_prompt},
            ],
            max_tokens=3000,
        )

        quality_report = response["choices"][0]["message"]["content"]

        return {
            "language_files": language_files,
            "language_loc": language_loc,
            "total_loc": total_loc,
            "quality_report": quality_report,
        }

    
    def analyze_code_complexity(self, contents: Dict[str, dict]) -> Dict:
        """
        Analyze code complexity metrics including cyclomatic complexity.

        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.

        Returns:
            Dict: Analysis results with complexity metrics.
        """
        complexity_metrics = {
            
            
            "cyclomatic_complexity": defaultdict(int),
        }

        # Collect metrics for each file
        for path, info in contents.items():
            if info["type"] == "file" and "language" in info:
                lang = info["language"]
                content = info["content"]

                # Analyze cyclomatic complexity (simplified example)
                complexity_metrics["cyclomatic_complexity"][lang] += self._calculate_cyclomatic_complexity(content, lang)

        return complexity_metrics

    

    def _calculate_cyclomatic_complexity(self, content: str, language: str) -> int:
        """
        Calculate cyclomatic complexity (simplified example).

        Args:
            content (str): The content of the file.
            language (str): The programming language.

        Returns:
            int: Estimated cyclomatic complexity.
        """
        # Simplified example: Count decision points
        if language == "ABAP":
            decision_keywords = ["IF", "ELSEIF", "CASE", "WHILE", "LOOP", "AND", "OR"]
        else:
            decision_keywords = ["if", "else", "case", "for", "while", "&&", "||"]
        
        return sum(content.count(keyword) for keyword in decision_keywords)

    def create_complexity_charts(self, complexity_metrics: Dict, repo_name: str) -> Dict[str, str]:
        """
        Create visualizations for code complexity metrics.

        Args:
            complexity_metrics (Dict): Dictionary containing complexity metrics.
            repo_name (str): Name of the repository.

        Returns:
            Dict[str, str]: Paths to the generated chart images.
        """
        chart_paths = {}

        try:
            # Import matplotlib with memory optimization
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import gc

            # Clear any existing plots
            plt.clf()
            plt.close('all')
            gc.collect()

            # Create a bar chart for cyclomatic complexity
            complexity_data = complexity_metrics.get("cyclomatic_complexity", {})
            if complexity_data:
                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)  # Reduced size and DPI
                
                languages = list(complexity_data.keys())
                values = list(complexity_data.values())
                
                bars = ax.bar(languages, values, color="lightgreen", edgecolor="black", linewidth=0.5)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
                
                ax.set_xlabel("Language", fontsize=11)
                ax.set_ylabel("Cyclomatic Complexity", fontsize=11)
                ax.set_title("Cyclomatic Complexity by Language", fontsize=12)
                
                # Rotate x-axis labels if needed
                if len(languages) > 3:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                
                plt.tight_layout()
                
                complexity_chart_path = self.output_dir / f"{repo_name}_complexity_chart.png"
                plt.savefig(complexity_chart_path, dpi=150, bbox_inches="tight",
                           facecolor='white', edgecolor='none')
                
                # Clean up
                plt.close(fig)
                plt.close('all')
                gc.collect()
                
                chart_paths["complexity"] = str(complexity_chart_path)

        except Exception as e:
            logging.error(f"Error generating complexity chart: {e}")
            # Clean up on error
            try:
                plt.close('all')
                gc.collect()
            except:
                pass

        return chart_paths
    
    
    def generate_documentation(self, github_link: str) -> Dict[str, str]:
        try:
            # Validate GitHub URL
            url_parse = self.validate_github_url(github_link)
            if not url_parse:
                raise ValueError("Invalid GitHub repository URL")

            username, repo_name = url_parse

            # Fetch repository contents
            contents = self.fetch_repository_contents(username, repo_name)

            # Generate directory tree if needed
            directory_tree = ""
            if self.config.include_code_structure:
                directory_tree = self._generate_directory_tree(contents)

            # Identify key file types
            language_counts = {}
            for path, info in contents.items():
                if info["type"] == "file" and "language" in info:
                    lang = info["language"]
                    language_counts[lang] = language_counts.get(lang, 0) + 1

            # Sort languages by frequency
            sorted_languages = sorted(
                language_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_languages = [lang for lang, count in sorted_languages[:5]]

            # Analyze top files if needed
            file_analyses = {}
            if self.config.max_files_to_analyze > 0:
                key_files = self.select_key_files(contents, self.config.max_files_to_analyze)
                file_analyses = self.analyze_files_parallel(key_files[:self.config.max_files_to_analyze])

            # Get code quality analysis if needed
            line_of_code = {}
            loc_chart_path = None
            if self.config.include_loc_analysis or self.config.include_loc_chart:
                line_of_code = self.analyze_code_quality(contents)
                if self.config.include_loc_chart:
                    loc_chart_path = self.create_lines_of_code_chart(
                        line_of_code["language_loc"], repo_name
                    )

            # Generate diagrams if needed
            diagrams = {}
            if any([
                self.config.include_class_diagram,
                self.config.include_flow_diagram,
                self.config.include_er_diagram,
                self.config.include_reference_architecture
            ]):
                diagrams = self.generate_diagrams_parallel(contents, repo_name)
                # Filter out diagrams that weren't requested
                if not self.config.include_class_diagram:
                    diagrams.pop("class_diagram", None)
                if not self.config.include_flow_diagram:
                    diagrams.pop("flow_diagram", None)
                if not self.config.include_er_diagram:
                    diagrams.pop("er_diagram", None)
                if not self.config.include_reference_architecture:
                    diagrams.pop("reference_architecture", None)
                
                # Log the diagrams that were generated for debugging
                logging.debug(f"Generated diagrams: {diagrams}")

            # Get code complexity analysis if needed
            complexity_metrics = {}
            complexity_charts = {}
            if self.config.include_complexity_analysis or self.config.include_complexity_charts:
                complexity_metrics = self.analyze_code_complexity(contents)
                if self.config.include_complexity_charts:
                    complexity_charts = self.create_complexity_charts(complexity_metrics, repo_name)

            # Extract SQL objects if needed
            sql_objects = {}
            if self.config.include_sql_objects and any(lang in ["SQL", "PL/SQL", "MySQL"] for lang in language_counts):
                sql_objects = self._extract_sql_objects(contents)

            # Generate project documentation using GPT-4-32k
            prompt = self._build_documentation_prompt(
                repo_name,
                directory_tree,
                top_languages,
                file_analyses,
                line_of_code,
                complexity_metrics,
                sql_objects
            )

            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior software architect with extensive expertise in analyzing and documenting codebases across a wide range of legacy and modern technologies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            documentation = response["choices"][0]["message"]["content"]

            # Save as markdown
            md_path = self.output_dir / f"{repo_name}_DOCUMENTATION.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(documentation)

            # Generate PDF with proper code block formatting for the tree
            pdf_path = self.markdown_to_pdf(
                documentation, 
                f"{repo_name}_DOCUMENTATION.pdf", 
                loc_chart_path,
                diagrams.get("class_diagram"),
                complexity_charts,
                sql_objects,
                contents,
                repo_name
            )

            # Prepare result dictionary with all generated artifacts
            result = {
                "markdown_path": str(md_path), 
                "pdf_path": str(pdf_path),
            }
            
            # Add diagram paths explicitly to ensure they're included
            if diagrams:
                # Explicitly add important diagrams with clear keys
                if "class_diagram" in diagrams and diagrams["class_diagram"] is not None:
                    result["class_diagram_path"] = str(diagrams["class_diagram"])
                if "reference_architecture" in diagrams and diagrams["reference_architecture"] is not None:
                    result["reference_architecture_path"] = str(diagrams["reference_architecture"])
                if "flow_diagram" in diagrams and diagrams["flow_diagram"] is not None:
                    result["flow_diagram_path"] = str(diagrams["flow_diagram"])
                if "er_diagram" in diagrams and diagrams["er_diagram"] is not None:
                    result["er_diagram_path"] = str(diagrams["er_diagram"])
                
            # Add chart paths if they were generated
            if loc_chart_path:
                result["loc_chart_path"] = str(loc_chart_path)
            if complexity_charts:
                result["complexity_charts"] = {
                    k: str(v) for k, v in complexity_charts.items()
                }
            
            # Log the final result for debugging
            logging.debug(f"Final result dictionary: {result}")

            return result

        except Exception as e:
            logging.error(f"Error in documentation generation: {e}")
            raise

    def generate_documentation_from_local(self, directory_path: str, repo_name: str) -> Dict[str, str]:
        """
        Generate documentation from a local directory containing source code.
        
        Args:
            directory_path (str): Path to the directory containing source code
            repo_name (str): Name to use for the repository
            
        Returns:
            Dict[str, str]: Dictionary of generated documentation paths
        """
        contents = {}
        
        try:
            # Normalize and validate directory path
            directory_path = os.path.abspath(directory_path)
            if not os.path.isdir(directory_path):
                raise ValueError(f"Invalid directory path: {directory_path}")

            # Walk through the directory structure
            for root, dirs, files in os.walk(directory_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(
                    fnmatch.fnmatch(os.path.join(root, d), pattern) 
                    for pattern in self.exclude_patterns
                )]

                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, directory_path)

                    # Skip excluded files
                    if any(fnmatch.fnmatch(rel_path, pattern) 
                        for pattern in self.exclude_patterns):
                        continue

                    # Get file extension and language
                    ext = os.path.splitext(file)[1][1:].lower()
                    if ext not in self.supported_extensions:
                        continue

                    try:
                        # Read file content with size check
                        file_size = os.path.getsize(file_path)
                        if file_size > self.max_file_size:
                            logging.warning(f"Skipping large file: {rel_path}")
                            continue

                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Add to contents structure
                        contents[rel_path] = {
                            "content": content,
                            "language": self.supported_extensions[ext],
                            "size": file_size,
                            "type": "file"
                        }

                    except UnicodeDecodeError:
                        logging.warning(f"Skipping binary file: {rel_path}")
                    except Exception as e:
                        logging.warning(f"Error processing {rel_path}: {str(e)}")

            # Generate documentation from the collected contents
            return self.generate_documentation_from_contents(contents, repo_name)

        except Exception as e:
            logging.error(f"Error processing local directory: {e}")
            raise        
            
    def _build_documentation_prompt(
        self,
        repo_name: str,
        directory_tree: str,
        top_languages: List[str],
        file_analyses: Dict[str, str],
        line_of_code: Dict,
        complexity_metrics: Dict,
        sql_objects: Dict
    ) -> str:
        """
        Build the documentation prompt based on the configuration options.
        """
        prompt_parts = []
        
        # Project Overview section
        if self.config.include_overview:
            prompt_parts.append(f"""
            ### 1. Project Overview
            - Core functionality and system architecture
            - Key features and capabilities
            """)
        
        # Executive Summary section
        if self.config.include_executive_summary:
            prompt_parts.append(f"""
            ### 2. Executive Summary
            - Business objectives and success metrics
            - Target users and stakeholders
            - Strategic value proposition
            """)
        
        # Technology Stack section
        if self.config.include_tech_stack:
            prompt_parts.append(f"""
            ### 3. Technology Stack
            Create a well-formatted table with the following structure:
            
            | Category | Technologies/Tools |
            |----------|-------------------|
            | Programming Languages | {', '.join(top_languages)} |
            | Frameworks | |
            | Databases | |
            | DevOps | |
            | APIs/Services | |
            """)
        
        # Code Structure section
        if self.config.include_code_structure:
            prompt_parts.append(f"""
            ### 4. Code Structure
            Present the code structure as a vertical directory tree exactly as follows:
            
            ```
            {directory_tree}
            ```
            
            Below the tree, include:
            - Key module responsibilities
            - Entry points and core components
            """)
        
        # Lines of Code Analysis section
        if self.config.include_loc_analysis:
            prompt_parts.append(f"""
            ### 5. Lines of Code Analysis
            - Breakdown by language (include visual representation in the final document)
            - Code quality score and metrics
            - Best practices implemented
            - Code issues and anti-patterns
            - Technical debt assessment
            """)
        
        # Code Complexity Analysis section
        if self.config.include_complexity_analysis:
            prompt_parts.append(f"""
            ### 6. Code Complexity Analysis
            - Cyclomatic complexity by language (include visual representation)
            Cyclomatic complexity: {complexity_metrics.get('cyclomatic_complexity', {})}
            - Recommendations for reducing complexity
            """)
        
        # Application Features section
        if self.config.include_features:
            prompt_parts.append(f"""
            ### 7. Application Features
            - Core functionality
            - User-facing features
            - Administrative capabilities
            - Performance characteristics
            - Include any unique or innovative features
            - Highlight integration points with third party applications
            """)
        
        # Dependencies section
        if self.config.include_dependencies:
            prompt_parts.append(f"""
            ### 8. Dependencies
            Create a table mapping pages/components to their API dependencies:
        
            | Page/Component | API Endpoint | Purpose | Request Method |
            |-----------------|--------------|---------|----------------|
            [Populate this table with:
            - Frontend page/component names
            - Corresponding API endpoints
            - Purpose of each API call
            - HTTP methods (GET/POST/PUT/DELETE)
            - Critical service dependencies and failure points
            - External system integrations and data flows
            - Internal module dependencies
            - External service dependencies
            - Third-party integrations
            """)
        
        # Known Issues & Challenges section
        if self.config.include_issues:
            prompt_parts.append(f"""
            ### 9. Known Issues & Challenges
            - Current limitations
            - Technical debt
            - Optimization opportunities
            - Missing Best Practices and Code Issues
            """)
        
        # SQL Objects section
        if self.config.include_sql_objects and sql_objects:
            prompt_parts.append(f"""
            ### 10. SQL Objects
            {sql_objects}
            """)
        
        # Combine all sections
        prompt = f"""
        Generate comprehensive documentation for {repo_name} GitHub repository based on the following sections.
        
        Repository structure summary:
        ```
        {directory_tree}
        ```
        
        Primary languages: {', '.join(top_languages)}
        
        Key file analyses:
        {file_analyses}
        
        Code quality analysis:
        {line_of_code.get('quality_report', '')}
        
        Please include ONLY these sections, with each section occupying approximately one page in length:
        """ + "\n".join(prompt_parts) + """
        
        Keep each section focused and concise, aiming for approximately one page per major section.
        """
        
        return prompt

    def _extract_sql_objects(self, contents: Dict[str, dict]) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract SQL objects (procedures, functions, triggers) from SQL/PLSQL files.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            
        Returns:
            Dict[str, List[Dict[str, str]]]: Dictionary containing lists of procedures, functions, and triggers.
        """
        sql_objects = {
            "procedures": [],
            "functions": [],
            "triggers": []
        }

        for path, info in contents.items():
            if info["type"] == "file" and info["language"] in ["SQL", "PL/SQL", "MySQL"]:
                content = info["content"]
                # Extract procedures
                procedures = re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([^\s(]+)", content, re.IGNORECASE)
                for proc in procedures:
                    proc_details = self._extract_sql_procedure_details(content, proc)
                    # Get the full SQL code for this procedure
                    full_proc_code = self._extract_full_sql_object(content, "PROCEDURE", proc)
                    sql_objects["procedures"].append({
                        "name": proc,
                        "file": path,
                        "details": proc_details,
                        "full_code": full_proc_code
                    })
                # Extract functions
                functions = re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([^\s(]+)", content, re.IGNORECASE)
                for func in functions:
                    func_details = self._extract_sql_function_details(content, func)
                    # Get the full SQL code for this function
                    full_func_code = self._extract_full_sql_object(content, "FUNCTION", func)
                    sql_objects["functions"].append({
                        "name": func,
                        "file": path,
                        "details": func_details,
                        "full_code": full_func_code
                    })
                # Extract triggers
                triggers = re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+([^\s(]+)", content, re.IGNORECASE)
                for trig in triggers:
                    trig_details = self._extract_sql_trigger_details(content, trig)
                    # Get the full SQL code for this trigger
                    full_trig_code = self._extract_full_sql_object(content, "TRIGGER", trig)
                    sql_objects["triggers"].append({
                        "name": trig,
                        "file": path,
                        "details": trig_details,
                        "full_code": full_trig_code
                    })

        return sql_objects

    def _extract_full_sql_object(self, content: str, object_type: str, object_name: str) -> str:
        """
        Extract the complete SQL code for a given object.
        
        Args:
            content (str): The content of the SQL file.
            object_type (str): The type of SQL object (PROCEDURE, FUNCTION, TRIGGER).
            object_name (str): The name of the SQL object.
            
        Returns:
            str: Complete SQL code for the object.
        """
        # Pattern to match the entire SQL object definition
        pattern = re.compile(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?{object_type}\s+{object_name}[\s\S]*?;(?:\s*/\s*)?",
            re.IGNORECASE | re.DOTALL
        )
        
        match = pattern.search(content)
        if match:
            return match.group(0).strip()
        return "Full code not found"

    def _extract_sql_procedure_details(self, content: str, procedure_name: str) -> Dict[str, str]:
        """
        Extract details of a SQL procedure.
        
        Args:
            content (str): The content of the SQL file.
            procedure_name (str): The name of the procedure.
            
        Returns:
            Dict[str, str]: Dictionary containing procedure details.
        """
        # Find the procedure definition with more flexible pattern matching
        proc_pattern = re.compile(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+{procedure_name}\s*\(([^)]*)\)\s*(?:AS|IS|BEGIN)[\s\S]*?(?:END\s*{procedure_name}\s*;|END;)",
            re.IGNORECASE | re.DOTALL
        )
        match = proc_pattern.search(content)
        
        # Return default values if no match is found
        if not match:
            return {
                "parameters": "N/A",
                "body": "N/A",
                "description": self._extract_comment_description(content, "PROCEDURE", procedure_name)
            }
        
        full_definition = match.group(0)
        params = match.group(1).strip() if match.group(1) else "N/A"
        
        # Extract the body between AS/IS/BEGIN and END
        body_pattern = re.compile(r"(?:AS|IS|BEGIN)([\s\S]*?)(?:END\s*\w*\s*;|END;)", re.IGNORECASE | re.DOTALL)
        body_match = body_pattern.search(full_definition)
        body = body_match.group(1).strip() if body_match else "N/A"
        
        # Extract additional attributes if present
        attributes = {}
        
        # Check for AUTHID attribute
        authid_match = re.search(r"AUTHID\s+(DEFINER|CURRENT_USER)", full_definition, re.IGNORECASE)
        if authid_match:
            attributes["authid"] = authid_match.group(1)
        
        # Check for PARALLEL_ENABLE attribute
        if re.search(r"PARALLEL_ENABLE", full_definition, re.IGNORECASE):
            attributes["parallel_enable"] = "Yes"
        
        # Check for DETERMINISTIC attribute
        if re.search(r"DETERMINISTIC", full_definition, re.IGNORECASE):
            attributes["deterministic"] = "Yes"
        
        # Get any comments/description preceding the procedure
        description = self._extract_comment_description(content, "PROCEDURE", procedure_name)
        
        return {
            "parameters": params,
            "body": body,
            "attributes": attributes,
            "description": description
        }

    def _extract_sql_function_details(self, content: str, function_name: str) -> Dict[str, str]:
        """
        Extract details of a SQL function.
        
        Args:
            content (str): The content of the SQL file.
            function_name (str): The name of the function.
            
        Returns:
            Dict[str, str]: Dictionary containing function details.
        """
        # Find the function definition with improved pattern matching
        func_pattern = re.compile(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+{function_name}\s*\(([^)]*)\)\s*RETURN\s+(\w+(?:\([^)]*\))?)[\s\S]*?(?:END\s*{function_name}\s*;|END;)",
            re.IGNORECASE | re.DOTALL
        )
        match = func_pattern.search(content)
        
        # Return default values if no match is found
        if not match:
            # Try alternative pattern (MySQL style)
            alt_func_pattern = re.compile(
                rf"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+{function_name}\s*\(([^)]*)\)\s*RETURNS\s+(\w+(?:\([^)]*\))?)[\s\S]*?(?:END\s*;|END)",
                re.IGNORECASE | re.DOTALL
            )
            match = alt_func_pattern.search(content)
            
            if not match:
                return {
                    "parameters": "N/A",
                    "return_type": "N/A",
                    "body": "N/A",
                    "description": self._extract_comment_description(content, "FUNCTION", function_name)
                }
        
        full_definition = match.group(0)
        params = match.group(1).strip() if match.group(1) else "N/A"
        return_type = match.group(2).strip() if match.group(2) else "N/A"
        
        # Extract the body between AS/IS/BEGIN and END
        body_pattern = re.compile(r"(?:AS|IS|BEGIN)([\s\S]*?)(?:END\s*\w*\s*;|END;)", re.IGNORECASE | re.DOTALL)
        body_match = body_pattern.search(full_definition)
        body = body_match.group(1).strip() if body_match else "N/A"
        
        # Extract additional attributes
        attributes = {}
        
        # Check for AUTHID attribute
        authid_match = re.search(r"AUTHID\s+(DEFINER|CURRENT_USER)", full_definition, re.IGNORECASE)
        if authid_match:
            attributes["authid"] = authid_match.group(1)
        
        # Check for DETERMINISTIC attribute
        if re.search(r"DETERMINISTIC", full_definition, re.IGNORECASE):
            attributes["deterministic"] = "Yes"
        
        # Check for PIPELINED attribute
        if re.search(r"PIPELINED", full_definition, re.IGNORECASE):
            attributes["pipelined"] = "Yes"
        
        # Check for PARALLEL_ENABLE attribute
        if re.search(r"PARALLEL_ENABLE", full_definition, re.IGNORECASE):
            attributes["parallel_enable"] = "Yes"
        
        # Get any comments/description preceding the function
        description = self._extract_comment_description(content, "FUNCTION", function_name)
        
        return {
            "parameters": params,
            "return_type": return_type,
            "body": body,
            "attributes": attributes,
            "description": description
        }

    def _extract_sql_trigger_details(self, content: str, trigger_name: str) -> Dict[str, str]:
        """
        Extract details of a SQL trigger.
        
        Args:
            content (str): The content of the SQL file.
            trigger_name (str): The name of the trigger.
            
        Returns:
            Dict[str, str]: Dictionary containing trigger details.
        """
        # Find the trigger definition with improved pattern matching for various SQL dialects
        # Oracle style
        oracle_trig_pattern = re.compile(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+{trigger_name}\s+(.*?)\s+ON\s+(\w+)(?:\.\w+)?\s*(.*?)(?:FOR\s+EACH\s+ROW\s*)?(?:WHEN\s*\(([^)]+)\)\s*)?(?:AS|IS|BEGIN)([\s\S]*?)(?:END\s*{trigger_name}\s*;|END;)",
            re.IGNORECASE | re.DOTALL
        )
        
        # MySQL style
        mysql_trig_pattern = re.compile(
            rf"CREATE\s+(?:DEFINER\s*=\s*[^\s]+\s+)?TRIGGER\s+{trigger_name}\s+(BEFORE|AFTER|INSTEAD\s+OF)\s+(INSERT|UPDATE|DELETE)(?:\s+OR\s+(INSERT|UPDATE|DELETE))?(?:\s+OR\s+(INSERT|UPDATE|DELETE))?\s+ON\s+(\w+)(?:\.\w+)?\s*(?:FOR\s+EACH\s+ROW\s*)?(?:WHEN\s*\(([^)]+)\)\s*)?(BEGIN[\s\S]*?END(?:\s*;)?)",
            re.IGNORECASE | re.DOTALL
        )
        
        # SQL Server style
        sqlserver_trig_pattern = re.compile(
            rf"CREATE\s+(?:OR\s+ALTER\s+)?TRIGGER\s+{trigger_name}\s+ON\s+(\w+)(?:\.\w+)?\s*(AFTER|INSTEAD\s+OF|FOR)\s+(INSERT|UPDATE|DELETE)(?:\s*,\s*(INSERT|UPDATE|DELETE))?(?:\s*,\s*(INSERT|UPDATE|DELETE))?\s*(?:WITH\s+[^(]+\s*)?(?:AS\s+)?(BEGIN[\s\S]*?END(?:\s*;)?)",
            re.IGNORECASE | re.DOTALL
        )
        
        # Try each pattern
        match = oracle_trig_pattern.search(content)
        if not match:
            match = mysql_trig_pattern.search(content)
        if not match:
            match = sqlserver_trig_pattern.search(content)
        
        # Return default values if no match is found with any pattern
        if not match:
            return {
                "trigger_type": "N/A",
                "table_name": "N/A",
                "trigger_condition": "N/A",
                "when_condition": "N/A",
                "body": "N/A",
                "description": self._extract_comment_description(content, "TRIGGER", trigger_name)
            }
        
        # Since we've matched different patterns, we need to extract the fields carefully
        full_definition = match.group(0)
        
        # Extract fields based on which pattern matched
        if oracle_trig_pattern.search(content):
            trigger_type = match.group(1).strip() if match.group(1) else "N/A"
            table_name = match.group(2).strip() if match.group(2) else "N/A"
            trigger_condition = match.group(3).strip() if match.group(3) else "N/A"
            when_condition = match.group(4).strip() if match.group(4) else "N/A"
            body = match.group(5).strip() if match.group(5) else "N/A"
        elif mysql_trig_pattern.search(content):
            trigger_type = match.group(1).strip() if match.group(1) else "N/A"
            trigger_condition = " OR ".join([g for g in [match.group(2), match.group(3), match.group(4)] if g])
            table_name = match.group(5).strip() if match.group(5) else "N/A"
            when_condition = match.group(6).strip() if match.group(6) and len(match.groups()) > 6 else "N/A"
            body = match.group(7).strip() if match.group(7) and len(match.groups()) > 7 else "N/A"
        elif sqlserver_trig_pattern.search(content):
            table_name = match.group(1).strip() if match.group(1) else "N/A"
            trigger_type = match.group(2).strip() if match.group(2) else "N/A"
            trigger_condition = " OR ".join([g for g in [match.group(3), match.group(4), match.group(5)] if g])
            when_condition = "N/A"  # SQL Server uses different syntax for conditional triggers
            body = match.group(6).strip() if match.group(6) and len(match.groups()) > 6 else "N/A"
        else:
            trigger_type = "N/A"
            table_name = "N/A"
            trigger_condition = "N/A"
            when_condition = "N/A"
            body = "N/A"
        
        # Extract additional attributes
        timing = "N/A"
        if "BEFORE" in full_definition:
            timing = "BEFORE"
        elif "AFTER" in full_definition:
            timing = "AFTER"
        elif "INSTEAD OF" in full_definition:
            timing = "INSTEAD OF"
        
        event = []
        if re.search(r"\b(INSERT)\b", full_definition, re.IGNORECASE):
            event.append("INSERT")
        if re.search(r"\b(UPDATE)\b", full_definition, re.IGNORECASE):
            event.append("UPDATE")
        if re.search(r"\b(DELETE)\b", full_definition, re.IGNORECASE):
            event.append("DELETE")
        
        # Check for FOR EACH ROW
        row_level = "Yes" if "FOR EACH ROW" in full_definition else "No"
        
        # Get any comments/description preceding the trigger
        description = self._extract_comment_description(content, "TRIGGER", trigger_name)
        
        return {
            "trigger_type": trigger_type,
            "table_name": table_name,
            "trigger_condition": trigger_condition,
            "when_condition": when_condition,
            "body": body,
            "timing": timing,
            "event": ", ".join(event) if event else "N/A",
            "row_level": row_level,
            "description": description
        }

    def _extract_comment_description(self, content: str, object_type: str, object_name: str) -> str:
        """
        Extract comments or descriptions that precede a SQL object.
        
        Args:
            content (str): The content of the SQL file.
            object_type (str): The type of SQL object (PROCEDURE, FUNCTION, TRIGGER).
            object_name (str): The name of the SQL object.
            
        Returns:
            str: The description/comment if found, otherwise "N/A".
        """
        # Look for a comment block before the object definition
        # Pattern to find position of object definition
        object_pattern = re.compile(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?{object_type}\s+{object_name}",
            re.IGNORECASE
        )
        object_match = object_pattern.search(content)
        
        if not object_match:
            return "N/A"
        
        # Get the position where the object definition starts
        start_pos = object_match.start()
        
        # Look for comments before the object definition
        # Try multi-line comment (/* */)
        multiline_comment_pattern = re.compile(
            r"/\*\s*([\s\S]*?)\s*\*/\s*$",
            re.MULTILINE
        )
        
        # Get the content before the object definition
        content_before = content[:start_pos].strip()
        
        # Try to find a multi-line comment right before the object
        multiline_match = multiline_comment_pattern.search(content_before)
        if multiline_match:
            return multiline_match.group(1).strip()
        
        # Try to find single-line comments (-- or #)
        lines_before = content_before.split('\n')
        comments = []
        
        # Start from the end and go backwards
        for line in reversed(lines_before):
            line = line.strip()
            if line.startswith('--') or line.startswith('#'):
                # Remove the comment marker and add to our list
                comment_text = line[2:] if line.startswith('--') else line[1:]
                comments.insert(0, comment_text.strip())
            elif not line:
                # Empty line is okay, continue collecting comments
                continue
            else:
                # Non-comment line found, stop collecting
                break
        
        if comments:
            return '\n'.join(comments)
        
        return "N/A"
            
    def _generate_directory_tree(
        self, contents: Dict[str, dict], prefix: str = "", depth: int = 3
    ) -> str:
        """
        Generate an ASCII directory tree from repository contents, limited by depth.
        Displays the structure in a vertical tree format that preserves formatting in markdown and PDF.

        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            prefix (str): Prefix for indentation in the tree structure.
            depth (int): Maximum depth to traverse in the directory tree.

        Returns:
            str: ASCII representation of the directory tree.
        """
        if depth < 0:
            return ""

        # Sort items by path for consistent ordering
        paths = sorted([path for path in contents.keys()])

        # Organize paths into a hierarchical structure
        tree_dict = {}
        for path in paths:
            parts = path.split(os.path.sep)
            current = tree_dict
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf node - store the full path
                    current.setdefault(part, {"__path": path})
                else:
                    # Directory node
                    current.setdefault(part, {})
                    current = current[part]

        # Generate tree string
        result = []

        

        def _build_tree(node, prefix="", is_last=True, level=0):
            if level > depth:
                return

            for i, (name, subtree) in enumerate(sorted(node.items())):
                if name == "__path":
                    continue

                is_last_item = i == len(node) - 1 or (
                    i == len(node) - 2 and "__path" in node
                )

                # Determine if the current node is a directory or a file
                is_directory = "__path" not in subtree
                icon = "📁" if is_directory else "📄"

                # Create proper tree connectors
                if is_last_item:
                    line = prefix + "└── " + icon + " " + name
                    next_prefix = prefix + "    "
                else:
                    line = prefix + "├── " + icon + " " + name
                    next_prefix = prefix + "│   "

                result.append(line)

                if subtree and name != "__path":
                    _build_tree(subtree, next_prefix, is_last_item, level + 1)

        _build_tree(tree_dict)
        return "\n".join(result)

    def markdown_to_pdf(
        self, markdown_text: str, filename: str, loc_chart_path: str = None, 
        class_diagram_path: str = None, complexity_charts: Dict[str, str] = None,
        sql_objects: Dict[str, List[Dict[str, str]]] = None, contents: Dict[str, dict] = None,
        repo_name: str = None
    ) -> str:
        try:
            output_path = self.output_dir / filename
            doc = SimpleDocTemplate(
                str(output_path), pagesize=letter, topMargin=36, bottomMargin=36
            )
            styles = getSampleStyleSheet()

            # Create custom styles for headings and content
            heading1_style = ParagraphStyle(
                "CustomHeading1",
                parent=styles["Heading1"],
                fontSize=20,
                textColor=colors.darkblue,
                spaceAfter=20,
                spaceBefore=20,
                fontName="Helvetica-Bold",
                keepWithNext=True,  # Ensures heading stays with following content
            )

            heading2_style = ParagraphStyle(
                "CustomHeading2",
                parent=styles["Heading2"],
                fontSize=16,
                textColor=colors.darkblue,
                spaceAfter=15,
                spaceBefore=15,
                fontName="Helvetica-Bold",
                keepWithNext=True,  # Ensures heading stays with following content
            )

            heading3_style = ParagraphStyle(
                "CustomHeading3",
                parent=styles["Heading3"],
                fontSize=14,
                textColor=colors.darkblue,
                spaceAfter=10,
                spaceBefore=10,
                fontName="Helvetica-Bold",
                keepWithNext=True,  # Ensures heading stays with following content
            )

            normal_style = ParagraphStyle(
                "CustomNormal",
                parent=styles["Normal"],
                fontSize=12,
                textColor=colors.black,
                spaceAfter=10,
                spaceBefore=10,
                leading=14,
            )

            code_style = ParagraphStyle(
                "CodeStyle",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=10,
                leading=12,
                leftIndent=20,
                rightIndent=20,
                backColor=colors.lightgrey,
                preserveLines=True,  # Important for preserving line breaks
            )

            bullet_style = ParagraphStyle(
                "BulletStyle",
                parent=styles["Normal"],
                fontSize=12,
                textColor=colors.black,
                spaceAfter=5,
                spaceBefore=5,
                leading=14,
                leftIndent=10,
                bulletIndent=0,
                bulletFontName="Helvetica",
                bulletFontSize=12,
            )

            nested_bullet_style = ParagraphStyle(
                "NestedBulletStyle",
                parent=styles["Normal"],
                fontSize=12,
                textColor=colors.black,
                spaceAfter=5,
                spaceBefore=5,
                leading=14,
                leftIndent=20,
                bulletIndent=10,
                bulletFontName="Helvetica",
                bulletFontSize=12,
            )

            tree_style = ParagraphStyle(
                "TreeStyle",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=10,
                leading=12,
                leftIndent=20,
                rightIndent=20,
                backColor=colors.lightgrey,
                preserveLines=True,  # Critical for tree display
                spaceAfter=0,
                spaceBefore=0,
            )

            # Enhanced table style for better readability
            table_style = TableStyle(
                [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),  # Professional blue header
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),  # White header text
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),  # Align all cells to the left
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),  # Header font
                    ("FONTSIZE", (0, 0), (-1, 0), 11),  # Slightly larger header font
                    ("FONTSIZE", (0, 1), (-1, -1), 10),  # Body font size
                    ("TOPPADDING", (0, 0), (-1, 0), 8),  # Header top padding
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),  # Header bottom padding
                    ("TOPPADDING", (0, 1), (-1, -1), 6),  # Body top padding
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 6),  # Body bottom padding
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),  # Light gray body
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),  # Subtle grid lines
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),  # Top vertical alignment for better text flow
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),  # Increased left padding
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),  # Increased right padding
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F8F8')]),  # Alternating row colors
                ]
            )

            story = []

            # Add a title to the PDF
            title = Paragraph("Project Documentation", heading1_style)
            story.append(title)
            story.append(Spacer(1, 24))

            # Add current date
            date_str = datetime.datetime.now().strftime("%B %d, %Y")
            date_para = Paragraph(f"Generated on: {date_str}", normal_style)
            story.append(date_para)
            story.append(Spacer(1, 24))

            # Add Table of Contents - dynamically based on config
            toc_data = [["Section", "Page"]]
            page_num = 1
            
            if self.config.include_overview:
                toc_data.append(["1. Project Overview", str(page_num)])
                page_num += 1
            if self.config.include_executive_summary:
                toc_data.append(["2. Executive Summary", str(page_num)])
                page_num += 1
            if self.config.include_tech_stack:
                toc_data.append(["3. Technology Stack", str(page_num)])
                page_num += 1
            if self.config.include_code_structure:
                toc_data.append(["4. Code Structure", str(page_num)])
                page_num += 1
            if self.config.include_loc_analysis:
                toc_data.append(["5. Lines of Code Analysis", str(page_num)])
                page_num += 1
            if self.config.include_complexity_analysis:
                toc_data.append(["6. Code Complexity Analysis", str(page_num)])
                page_num += 1
            if self.config.include_features:
                toc_data.append(["7. Application Features", str(page_num)])
                page_num += 1
            if self.config.include_dependencies:
                toc_data.append(["8. Dependencies", str(page_num)])
                page_num += 1
            if self.config.include_issues:
                toc_data.append(["9. Known Issues & Challenges", str(page_num)])
                page_num += 1
            if self.config.include_sql_objects and sql_objects:
                toc_data.append(["10. SQL Objects", str(page_num)])
                page_num += 1
            
            # Include line-by-line documentation if enabled
            if self.config.include_line_by_line_docs:
                toc_data.append(["Explain Code Line by Line", str(page_num)])
                page_num += 1

            toc_table = Table(toc_data, colWidths=[400, 100])
            toc_table.setStyle(table_style)
            story.append(Paragraph("Table of Contents", heading2_style))
            story.append(Spacer(1, 12))
            story.append(toc_table)
            story.append(Spacer(1, 12))
            story.append(PageBreak())

            # Clean markdown text by removing ## headers and improving alignment
            cleaned_markdown = self._clean_markdown_content(markdown_text)
            
            # Convert markdown to HTML and process content
            html_content = markdown.markdown(
                cleaned_markdown, extensions=["tables", "fenced_code"]
            )

            # Special handling for code blocks containing the directory tree
            # First, extract code blocks from the markdown before conversion
            code_blocks = re.findall(r"```(?:\w+)?\n([\s\S]+?)\n```", markdown_text)
            tree_blocks = []

            # Identify tree-like code blocks (those with indentation and tree characters)
            for block in code_blocks:
                if any(marker in block for marker in ["├──", "└──", "│", "─"]):
                    tree_blocks.append(block)

            # Create a set to track processed tree blocks
            processed_tree_blocks = set()

            # Process content section by section
            sections = re.split(r"<h[1-6]>", html_content)
            current_section = 0

            for section in sections:
                if not section.strip():
                    continue

                # Extract heading level and content
                heading_match = re.match(r"(.+?)</h[1-6]>", section)
                if heading_match:
                    heading_text = heading_match.group(1).strip()
                    content = section[heading_match.end() :].strip()

                    # Add page break before each main section (except the first one)
                    if (
                        heading_text.startswith(
                            ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8", "9")
                        )
                        and current_section > 0
                    ):
                        story.append(PageBreak())

                    current_section += 1

                    # Determine heading style based on heading level
                    if heading_text.startswith("1."):
                        story.append(Paragraph(heading_text, heading1_style))
                    elif heading_text.startswith("2."):
                        story.append(Paragraph(heading_text, heading2_style))
                    elif heading_text.startswith("3."):
                        story.append(Paragraph(heading_text, heading3_style))
                    else:
                        story.append(Paragraph(heading_text, heading2_style))

                    # Special handling for Technology Stack section to properly format the table
                    if heading_text.startswith("3. Technology Stack"):
                        # Extract table data from the content
                        table_html = (
                            content[
                                content.find("<table>") : content.find("</table>") + 8
                            ]
                            if "<table>" in content
                            else ""
                        )
                        table_data = (
                            self._html_table_to_data(table_html) if table_html else []
                        )

                        if table_data:
                            # Calculate dynamic column widths based on content
                            if len(table_data[0]) == 2:  # Two column table
                                col_widths = [150, 350]  # Category, Technologies
                            elif len(table_data[0]) == 3:  # Three column table
                                col_widths = [120, 200, 180]  # Adjust for 3 columns
                            elif len(table_data[0]) == 4:  # Four column table
                                col_widths = [120, 150, 120, 110]  # Adjust for 4 columns
                            else:
                                # Default equal width distribution
                                available_width = 500
                                col_widths = [available_width // len(table_data[0])] * len(table_data[0])

                            # Wrap cell text to fit within columns with better formatting
                            for i in range(len(table_data)):
                                for j in range(len(table_data[i])):
                                    if i == 0:  # Header row
                                        # Keep headers as plain text for better styling
                                        continue
                                    else:  # Data rows
                                        # Wrap long text in paragraphs
                                        cell_text = str(table_data[i][j])
                                        if len(cell_text) > 50:  # Only wrap long text
                                            table_data[i][j] = Paragraph(cell_text, normal_style)
                                        # Keep short text as is for better alignment

                            # Create the table with dynamic column widths
                            tech_table = Table(table_data, colWidths=col_widths)
                            tech_table.setStyle(table_style)

                            story.append(Spacer(1, 12))
                            story.append(tech_table)
                            story.append(Spacer(1, 12))

                            # Replace in content to avoid duplication
                            content = (
                                content.replace(table_html, "")
                                if table_html
                                else content
                            )

                    # Check if this is the Code Structure section
                    if heading_text.startswith("4. Code Structure") and tree_blocks:
                        story.append(Paragraph("Directory Structure:", normal_style))
                        story.append(Spacer(1, 8))

                        # Create a background container for the tree
                        story.append(Spacer(1, 10))

                        # Process tree line by line to maintain vertical format
                        for tree in tree_blocks:
                            tree = tree.replace("&lt;", "<").replace("&gt;", ">")
                            lines = tree.split("\n")

                            # Process each line of the tree separately
                            for line in lines:
                                if line.strip():
                                    # Preserve monospace formatting and all whitespace
                                    para = Paragraph(
                                        f'<font face="Courier" size="10"><xpre>{line}</xpre></font>',
                                        tree_style,
                                    )
                                    story.append(para)

                            # Mark this tree block as processed
                            processed_tree_blocks.add(tree)

                        story.append(Spacer(1, 12))

                        # Remove all code blocks containing tree structures from this section's content
                        for tree in tree_blocks:
                            html_encoded_tree = tree.replace("<", "&lt;").replace(
                                ">", "&gt;"
                            )
                            content = content.replace(
                                f"<pre><code>{html_encoded_tree}</code></pre>", ""
                            )

                    # Add the LOC chart in its section
                if self.config.include_loc_chart and loc_chart_path:
                    if (
                        heading_text.startswith("5. Lines of Code Analysis")
                        and loc_chart_path
                    ):
                        # story.append(PageBreak())
                        # story.append(
                        #     Paragraph("Lines of Code Visualization", heading2_style))
                        story.append(Spacer(1, 12))

                        # Add the image with proper sizing
                        img = Image(loc_chart_path, width=450, height=300)
                        story.append(img)
                        story.append(Spacer(1, 12))
                    pass

                if self.config.include_complexity_charts and complexity_charts:    
                    # Add complexity charts in their section - MOVED THIS INSIDE THE LOOP
                    if heading_text.startswith("6. Code Complexity Analysis") and complexity_charts:
                        if complexity_charts:
                            # Add cyclomatic complexity chart
                            if "complexity" in complexity_charts:
                                story.append(Paragraph("Cyclomatic Complexity by Language", heading2_style))
                                story.append(Spacer(1, 12))
                                img = Image(complexity_charts["complexity"], width=450, height=300)
                                story.append(img)
                                story.append(Spacer(1, 12))
                    pass            

                    # Process remaining content
                    # First, handle unordered lists (bullet points) - this is the main fix
                    if "<ul>" in content:
                        # Process all unordered lists
                        ul_pattern = re.compile(r"<ul>(.+?)</ul>", re.DOTALL)
                        ul_matches = ul_pattern.finditer(content)

                        for ul_match in ul_matches:
                            ul_content = ul_match.group(1)
                            # Find all list items in this unordered list
                            li_pattern = re.compile(r"<li>(.+?)</li>", re.DOTALL)
                            li_matches = li_pattern.finditer(ul_content)

                            for li_match in li_matches:
                                li_content = li_match.group(1)
                                # Check if this list item contains a nested list
                                if "<ul>" in li_content:
                                    # First add the parent bullet
                                    parent_text = li_content[
                                        : li_content.find("<ul>")
                                    ].strip()
                                    if parent_text:
                                        story.append(
                                            Paragraph(f"• {parent_text}", bullet_style)
                                        )

                                    # Then add the nested bullets
                                    nested_ul_content = li_content[
                                        li_content.find("<ul>") : li_content.find(
                                            "</ul>"
                                        )
                                        + 5
                                    ]
                                    nested_li_pattern = re.compile(
                                        r"<li>(.+?)</li>", re.DOTALL
                                    )
                                    nested_li_matches = nested_li_pattern.finditer(
                                        nested_ul_content
                                    )

                                    for nested_li_match in nested_li_matches:
                                        nested_text = nested_li_match.group(1).strip()
                                        if nested_text and not re.search(
                                            r"<ul>|</ul>", nested_text
                                        ):
                                            story.append(
                                                Paragraph(
                                                    f"  ○ {nested_text}",
                                                    nested_bullet_style,
                                                )
                                            )
                                else:
                                    # Regular bullet point
                                    story.append(
                                        Paragraph(f"• {li_content}", bullet_style)
                                    )

                            # Remove the processed unordered list from content
                            content = content.replace(ul_match.group(0), "")

                    # Handle code blocks in the remaining content
                    code_blocks = re.findall(
                        r"<pre><code>(.+?)</code></pre>", content, re.DOTALL
                    )
                    for code_block in code_blocks:
                        clean_code = re.sub(r"<[^>]+>", "", code_block)

                        # Skip tree-like code blocks that have already been processed
                        if any(
                            marker in clean_code for marker in ["├──", "└──", "│", "─"]
                        ):
                            # Check if this tree block was already processed
                            original_tree = clean_code.replace("&lt;", "<").replace(
                                "&gt;", ">"
                            )
                            if original_tree in processed_tree_blocks or any(
                                tree in original_tree for tree in processed_tree_blocks
                            ):
                                content = content.replace(
                                    f"<pre><code>{code_block}</code></pre>", "", 1
                                )
                                continue  # Skip this tree entirely

                            for line in clean_code.split("\n"):
                                if line.strip():
                                    story.append(
                                        Paragraph(
                                            f"<font face='Courier' size='10'><xpre>{line}</xpre></font>",
                                            code_style,
                                        )
                                    )
                        else:
                            story.append(Paragraph(clean_code, code_style))

                        story.append(Spacer(1, 12))
                        # Replace in content to avoid duplication
                        content = content.replace(
                            f"<pre><code>{code_block}</code></pre>", "", 1
                        )

                    # Handle tables (for tables other than Technology Stack)
                    if heading_text != "3. Technology Stack" and "<table>" in content:
                        table_html = content[
                            content.find("<table>") : content.find("</table>") + 8
                        ]
                        table_data = self._html_table_to_data(table_html)
                        if table_data:
                            # Calculate dynamic column widths
                            num_cols = len(table_data[0]) if table_data else 1
                            if num_cols == 2:
                                col_widths = [200, 300]
                            elif num_cols == 3:
                                col_widths = [150, 175, 175]
                            elif num_cols == 4:
                                col_widths = [125, 125, 125, 125]
                            elif num_cols == 5:
                                col_widths = [100, 100, 100, 100, 100]
                            else:
                                # Default equal distribution
                                available_width = 500
                                col_widths = [available_width // num_cols] * num_cols
                            
                            # Wrap cell text for better formatting
                            for i in range(len(table_data)):
                                for j in range(len(table_data[i])):
                                    if i == 0:  # Header row - keep as text
                                        continue
                                    else:  # Data rows
                                        cell_text = str(table_data[i][j])
                                        if len(cell_text) > 40:  # Wrap longer text
                                            table_data[i][j] = Paragraph(cell_text, normal_style)

                            # Create table with calculated column widths
                            table = Table(table_data, colWidths=col_widths)
                            table.setStyle(table_style)
                            story.append(table)
                            story.append(Spacer(1, 12))
                        # Replace in content to avoid duplication
                        content = content.replace(table_html, "")

                    # Process any remaining paragraphs
                    paragraphs = re.split(r"<br\s*/?>|</p>", content)
                    for para in paragraphs:
                        # Skip empty or already processed content
                        if not para.strip() or "<ul>" in para or "<li>" in para:
                            continue

                        # Clean text of HTML tags
                        clean_text = re.sub(r"<[^>]+>", "", para)
                        if clean_text.strip():
                            story.append(Paragraph(clean_text, normal_style))
                            story.append(Spacer(1, 8))

            # Add line-by-line code explanation section if enabled
            if self.config.include_line_by_line_docs:
                story.append(PageBreak())
                story.append(Paragraph("Explain Code Line by Line", heading1_style))
                story.append(Spacer(1, 12))
                
                # Generate line-by-line explanations for key files
                line_explanations = self.generate_line_by_line_explanations(contents or {}, repo_name or 'unknown')
                
                for file_path, explanation in line_explanations.items():
                    story.append(Paragraph(f"File: {file_path}", heading2_style))
                    story.append(Spacer(1, 8))
                    
                    # Split explanation into sections
                    sections = explanation.split('\n\n')
                    for section in sections:
                        if section.strip():
                            # Check if it's a code line (starts with line number)
                            if re.match(r'^\d+\.\d+', section.strip()):
                                # Format as code with explanation
                                story.append(Paragraph(section.strip(), code_style))
                            else:
                                # Format as normal text
                                story.append(Paragraph(section.strip(), normal_style))
                            story.append(Spacer(1, 6))
                    
                    story.append(Spacer(1, 12))

            # Simplified SQL objects section with only requested details
            if self.config.include_sql_objects and sql_objects:
                if sql_objects and (sql_objects["procedures"] or sql_objects["functions"] or sql_objects["triggers"]):
                    story.append(PageBreak())
                    story.append(Paragraph("10. SQL Objects", heading1_style))
                    story.append(Spacer(1, 12))

                    # Add procedures table - only name, parameters and description
                    if sql_objects["procedures"]:
                        story.append(Paragraph("10.1 Stored Procedures", heading2_style))
                        story.append(Spacer(1, 12))
                        
                        # Create a table with just the requested columns
                        proc_data = [["Name", "Parameters", "Description"]]
                        
                        for proc in sql_objects["procedures"]:
                            details = proc.get("details", {})
                            proc_data.append([
                                proc.get("name", "N/A"),
                                details.get("parameters", "N/A"),
                                details.get("description", "N/A")
                            ])
                                    
                        # Wrap long text in table cells
                        for i in range(1, len(proc_data)):  # Skip header row
                            for j in range(len(proc_data[i])):
                                if len(str(proc_data[i][j])) > 50:
                                    proc_data[i][j] = Paragraph(str(proc_data[i][j]), normal_style)
                        
                        proc_table = Table(proc_data, colWidths=[120, 180, 200])
                        proc_table.setStyle(table_style)
                        story.append(proc_table)
                        story.append(Spacer(1, 24))

                    # Add functions table - only name, parameters, return type and description
                    if sql_objects["functions"]:
                        story.append(Paragraph("10.2 Functions", heading2_style))
                        story.append(Spacer(1, 12))
                        
                        # Create a table with just the requested columns
                        func_data = [["Name", "Parameters", "Return Type", "Description"]]
                        
                        for func in sql_objects["functions"]:
                            details = func.get("details", {})
                            func_data.append([
                                func.get("name", "N/A"),
                                details.get("parameters", "N/A"),
                                details.get("return_type", "N/A"),
                                details.get("description", "N/A")
                            ])
                                    
                        # Wrap long text in table cells
                        for i in range(1, len(func_data)):  # Skip header row
                            for j in range(len(func_data[i])):
                                if len(str(func_data[i][j])) > 40:
                                    func_data[i][j] = Paragraph(str(func_data[i][j]), normal_style)
                        
                        func_table = Table(func_data, colWidths=[120, 130, 90, 160])
                        func_table.setStyle(table_style)
                        story.append(func_table)
                        story.append(Spacer(1, 24))

                    # Add triggers table - only name, table name, timing, event and description
                    if sql_objects["triggers"]:
                        story.append(Paragraph("10.3 Triggers", heading2_style))
                        story.append(Spacer(1, 12))
                        
                        # Create a table with just the requested columns
                        trig_data = [["Name", "Table Name", "Timing", "Event", "Description"]]
                        
                        for trig in sql_objects["triggers"]:
                            details = trig.get("details", {})
                            trig_data.append([
                                trig.get("name", "N/A"),
                                details.get("table_name", "N/A"),
                                details.get("timing", "N/A"),
                                details.get("event", "N/A"),
                                details.get("description", "N/A")
                            ])
                                    
                        # Wrap long text in table cells
                        for i in range(1, len(trig_data)):  # Skip header row
                            for j in range(len(trig_data[i])):
                                if len(str(trig_data[i][j])) > 30:
                                    trig_data[i][j] = Paragraph(str(trig_data[i][j]), normal_style)
                        
                        trig_table = Table(trig_data, colWidths=[120, 70, 70, 100, 140])
                        trig_table.setStyle(table_style)
                        story.append(trig_table)
                        story.append(Spacer(1, 24))
                pass    

            # Build the PDF
            doc.build(story)
            return str(output_path)

        except Exception as e:
            logging.error(f"Error generating PDF: {e}")
            raise

    def _is_tree_structure(self, text: str) -> bool:
        """
        Determine if the text is a directory tree structure.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if the text appears to be a directory tree, False otherwise.
        """
        # Count the occurrences of typical tree characters
        tree_markers = ["├──", "└──", "│", "─", "📁", "📄"]
        marker_count = sum(text.count(marker) for marker in tree_markers)

        # Check for indentation patterns
        lines = text.split("\n")
        indented_lines = sum(
            1 for line in lines if line.startswith("    ") or line.startswith("│   ")
        )

        # It's likely a tree if it has tree markers and indented lines
        return marker_count > 0 and indented_lines > 0

    def _format_tree_for_pdf(
        self, tree_content: str, story: list, tree_style: ParagraphStyle
    ) -> None:
        """
        Format a directory tree structure for PDF output with proper spacing and alignment.

        Args:
            tree_content (str): The tree content to format.
            story (list): The story list to append formatted content to.
            tree_style (ParagraphStyle): The style to apply to the tree.
        """
        # Replace HTML entities with actual characters
        tree_content = tree_content.replace("&lt;", "<").replace("&gt;", ">")

        # Start a container for the tree with background color
        story.append(
            Paragraph(
                "<para backColor=lightgrey borderColor=grey borderWidth=1 borderPadding=10>",
                ParagraphStyle("Container"),
            )
        )

        # Add each line of the tree with preserved formatting
        for line in tree_content.split("\n"):
            if line.strip():
                # Use monospaced font for tree structure
                story.append(
                    Paragraph(
                        f"<font face='Courier' size='10'><xpre>{line}</xpre></font>",
                        tree_style,
                    )
                )

        # Close the container
        story.append(Paragraph("</para>", ParagraphStyle("Container")))

        # Add space after the tree
        story.append(Spacer(1, 12))

    def _is_tree_already_processed(self,tree_content, processed_trees):
        """Check if a tree or a similar variant is already processed."""
        normalized_tree = "".join(tree_content.split())  # Remove all whitespace
        for processed_tree in processed_trees:
            normalized_processed = "".join(processed_tree.split())
            # If 90% of the content is the same, consider it a match
            if len(normalized_tree) > 0 and len(normalized_processed) > 0:
                similarity = sum(
                    a == b for a, b in zip(normalized_tree, normalized_processed)
                ) / max(len(normalized_tree), len(normalized_processed))
                if similarity > 0.9:
                    return True
        return False
    
    

    def _clean_markdown_content(self, markdown_text: str) -> str:
        """
        Clean markdown content by removing ## headers and improving alignment.
        
        Args:
            markdown_text (str): Original markdown text
            
        Returns:
            str: Cleaned markdown text with proper formatting
        """
        try:
            # Remove ## from headers while preserving the text and hierarchy
            cleaned_text = re.sub(r'^##\s+(.+)$', r'\1', markdown_text, flags=re.MULTILINE)
            
            # Ensure proper spacing after all headers
            cleaned_text = re.sub(r'^(#{1,6})\s*(.+)$', r'\1 \2', cleaned_text, flags=re.MULTILINE)
            
            # Improve list formatting with proper indentation
            cleaned_text = re.sub(r'^([*-])\s+(.+)$', r'\1 \2', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^\s+([*-])\s+(.+)$', r'  \1 \2', cleaned_text, flags=re.MULTILINE)
            
            # Ensure proper spacing around tables
            cleaned_text = re.sub(r'\n(\|.+\|)\n', r'\n\n\1\n\n', cleaned_text)
            
            # Clean up multiple consecutive newlines but preserve intentional spacing
            cleaned_text = re.sub(r'\n{4,}', '\n\n\n', cleaned_text)
            
            # Ensure proper spacing around code blocks
            cleaned_text = re.sub(r'(?<!\n)\n```', '\n\n```', cleaned_text)
            cleaned_text = re.sub(r'```\n(?!\n)', '```\n\n', cleaned_text)
            
            # Fix paragraph spacing
            cleaned_text = re.sub(r'\n([A-Z][^\n]*[.!?])\n([A-Z])', r'\n\1\n\n\2', cleaned_text)
            
            # Ensure consistent bullet point formatting
            cleaned_text = re.sub(r'^\s*[•·]\s+', '• ', cleaned_text, flags=re.MULTILINE)
            
            return cleaned_text.strip()
            
        except Exception as e:
            logging.error(f"Error cleaning markdown content: {e}")
            return markdown_text
    
    def _html_table_to_data(self, table_html: str) -> List[List[str]]:
        """
        Convert HTML table to a list of lists for ReportLab Table.
        
        Args:
            table_html (str): HTML table string.
            
        Returns:
            List[List[str]]: Table data as a list of rows.
        """
        if not table_html:
            return []
            
        try:
            # Use regex to extract table data
            rows = re.findall(r"<tr.*?>(.+?)</tr>", table_html, re.DOTALL)
            
            table_data = []
            for row in rows:
                # Check if this is a header row
                if "<th" in row:
                    cells = re.findall(r"<th.*?>(.+?)</th>", row, re.DOTALL)
                else:
                    cells = re.findall(r"<td.*?>(.+?)</td>", row, re.DOTALL)
                    
                # Strip HTML tags from cell content
                cleaned_cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
                
                if cleaned_cells:  # Only add non-empty rows
                    table_data.append(cleaned_cells)
                    
            return table_data
        except Exception as e:
            logging.error(f"Error parsing HTML table: {e}")
            return []

    def sample_usage(self):
        """
        Example of how to use this class.
        """
        # Example usage:
        github_link = "username/repository"
        result = self.generate_documentation(github_link)
        print(f"Documentation generated successfully:")
        print(f"- Markdown: {result['markdown_path']}")
        print(f"- PDF: {result['pdf_path']}")

    def create_lines_of_code_chart(self, language_loc: Dict[str, int], chartname) -> str:
        """
        Create a visually impressive vertical bar chart visualization for lines of code by language.

        Args:
            language_loc (Dict[str, int]): Dictionary mapping languages to lines of code.

        Returns:
            str: Path to the saved chart image.
        """
        try:
            # Use the 'Agg' backend with memory optimization
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import gc

            # Clear any existing plots to free memory
            plt.clf()
            plt.close('all')
            gc.collect()

            # Sort languages by lines of code (descending)
            sorted_langs = sorted(
                language_loc.items(), key=lambda x: x[1], reverse=True
            )

            # Limit to top 8 languages for memory efficiency
            top_langs = sorted_langs[:8]

            langs = [x[0] for x in top_langs]
            locs = [x[1] for x in top_langs]

            # Create smaller figure to reduce memory usage
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)  # Reduced DPI

            # Use a custom list of colors for the bars
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
            ]

            # Create vertical bar chart
            bars = ax.bar(langs, locs, color=colors[:len(langs)], edgecolor="black", linewidth=0.5)

            # Add data labels with formatting (simplified)
            max_loc = max(locs) if locs else 0
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (max_loc * 0.01),
                    f"{height:,}",
                    va="bottom", ha="center",
                    fontsize=9, fontweight="bold"
                )

            # Add a grid for better readability
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Remove spines (top and right)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set chart title and labels
            ax.set_xlabel("Programming Languages", fontsize=12, fontweight="bold")
            ax.set_ylabel("Lines of Code", fontsize=12, fontweight="bold")

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.tick_params(axis='y', labelsize=10)

            # Tight layout for better spacing
            plt.tight_layout()

            # Save the chart to a file with optimized settings
            chart_path = os.path.join(self.output_dir, f"{chartname}loc_chart.png")
            plt.savefig(chart_path, dpi=150, bbox_inches="tight", 
                       facecolor='white', edgecolor='none')
            
            # Clean up to free memory
            plt.close(fig)
            plt.close('all')
            gc.collect()

            return chart_path

        except Exception as e:
            logging.error(f"Error generating LOC chart: {e}")
            # Clean up on error
            try:
                plt.close('all')
                gc.collect()
            except:
                pass
            return None

    def generate_class_diagram(self, contents: Dict[str, dict], repo_name: str) -> str:
        """
        Generate UML class diagrams, flow diagrams, or ER diagrams based on the repository's primary language.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            repo_name (str): Name of the repository.
            
        Returns:
            str: Path to the generated diagram in Mermaid format.
        """
        if not self.config.include_class_diagram:
            return None
    
        try:
            # Determine the primary language of the repository
            primary_language = self._determine_primary_language(contents)
            
            if primary_language in ["Java", "C++", "C#", "Python", "VB.NET", "ASP.NET Razor", "C Header", "JavaScript", "HTML", "CSS","Visual Basic .NET",".NET Assembly","Text","XML","ABAP"]:
                # Generate UML Class Diagram for OOPS languages
                return self._generate_uml_class_diagram(contents, repo_name)
            
            elif primary_language in ["C", "COBOL", "RPG", "RPG ILE", "SQL RPG", "CL Program", "CL ILE", "DDS Source", "Display File", "Printer File", "Logical File", "Physical File"]:
                # Generate Flow Diagram for legacy languages and DDS files
                return self._generate_flow_diagram(contents, repo_name)
            
            elif primary_language in ["SQL", "PL/SQL", "MySQL"]:
                # Generate ER Diagram for SQL-based languages
                return self._generate_er_diagram(contents, repo_name)
            
            else:
                logging.warning(f"Unsupported primary language: {primary_language}")
                return None

        except Exception as e:
            logging.error(f"Error generating diagram: {e}")
            return f"Error generating diagram: {str(e)}"

    def _determine_primary_language(self, contents: Dict[str, dict]) -> str:
        """
        Determine the primary language of the repository based on file extensions.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            
        Returns:
            str: Primary language of the repository.
        """
        language_counts = {}
        
        for path, info in contents.items():
            if info["type"] == "file" and "language" in info:
                lang = info["language"]
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Special handling for COBOL - it often has fewer but larger files
        if "COBOL" in language_counts:
            # Calculate total lines of COBOL code
            cobol_loc = sum(
                len(info["content"].split("\n"))
                for path, info in contents.items()
                if info.get("language") == "COBOL"
            )
            # If COBOL has significant LOC, prioritize it
            if cobol_loc > 1000:  # At least 1000 lines of COBOL
                return "COBOL"
        
        # Return the language with the highest count
        return max(language_counts, key=language_counts.get, default="Unknown")

    def _generate_uml_class_diagram(self, contents: Dict[str, dict], repo_name: str) -> str:
        try:
            # Collect classes and their relationships from supported OOPS languages
            classes_by_language = {}
           
            # Filter for files in supported OOPS languages
            supported_languages = {
                "Java": [".java"],
                "C++": [".cpp", ".hpp", ".cc", ".cxx", ".h"],
                "C": [".c", ".h"],
                "Python": [".py"],
                ".NET Framework": [".cs", ".vb", ".fs"],
                "ABAP": [".abap"]  # Add ABAP support
            }
           
            # Collect files by language
            language_files = {}
            for lang, extensions in supported_languages.items():
                language_files[lang] = []
                for path, info in contents.items():
                    if info["type"] == "file" and any(path.endswith(ext) for ext in extensions):
                        language_files[lang].append((path, info))
           
            # Process each language with language-specific extractors
            for language, files in language_files.items():
                if not files:
                    continue
               
                # Take a representative sample of files to avoid overwhelming the API
                sample_files = files[:min(10, len(files))]
               
                # Extract code snippets for analysis
                code_samples = []
                for path, info in sample_files:
                    # Sanitize content to remove single and double quotes
                    sanitized_content = info["content"][:8000].replace("'", "").replace('"', "")
                    code_samples.append({
                        "path": path,
                        "content": sanitized_content  # Use sanitized content
                    })
               
                # Generate class extraction prompt based on language
                prompt = self._create_class_extraction_prompt(language, code_samples)
               
                # Use GPT-4 to extract class information
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert software architect specializing in analyzing code structure and generating UML class diagrams. Extract class information accurately for the requested language. Do not include single or double quotes in class names, attributes, or methods."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.2
                )
               
                class_data = response["choices"][0]["message"]["content"]
                classes_by_language[language] = class_data
           
            # Combine class data from all languages into a unified diagram
            combined_prompt = f"""
            Create a unified UML class diagram in Mermaid syntax based on the following class information extracted from different languages:
            {classes_by_language}
            Rules for the diagram:
                Create a Mermaid class diagram with the following specifications:
                    1. Focus on the most important classes (maximum 15-20 classes total for readability)
                    2. Include class attributes and methods for each class (use the most significant ones)
                    3. Show relationships using single-word labels only:
                    - Use "extends" for inheritance
                    - Use "contains" for composition
                    - Use "has" for association
                    4. Group classes by component/module when possible
                    5. Use proper Mermaid classDiagram syntax
                    6. Include language-specific notation where appropriate (e.g., static, abstract)
                    7. Keep the diagram clear and readable
                    8. Do not include single or double quotes in class names, attributes, or methods
                    9. Do not use any special characters
                    10. Do not use dot (.) for any character, use it as is
                    11. Treat everything as a class
                    12. dot seprateword by dot (.) take only last word
                    13. do not include empty classes.
                    14. do not include special symbols
 
                    The diagram should model [DESCRIBE YOUR SYSTEM HERE - e.g., "an e-commerce platform", "a hospital management system", etc.] with appropriate modules for [LIST KEY COMPONENTS - e.g., "user management, product catalog, order processing", etc.].
 
                    Please ensure the diagram is properly formatted with correct syntax and maintains clear, logical relationships between classes.
                           
            """
           
            # Generate the unified Mermaid class diagram
            diagram_response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in analyzing legacy code and generating class diagrams in Mermaid syntax.. Do not include single or double quotes in class names, attributes, or methods or take every thing as class."
                    },
                    {"role": "user", "content": combined_prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
           
            mermaid_diagram = diagram_response["choices"][0]["message"]["content"]
           
            # Extract the diagram code from the response
            diagram_code_match = re.search(r"```mermaid\n(.*?)```", mermaid_diagram, re.DOTALL)
            diagram_code = diagram_code_match.group(1) if diagram_code_match else mermaid_diagram
           
            # Post-process the diagram code to remove any remaining quotes
            diagram_code = diagram_code.replace("'", "").replace('"', "")
           
            # Save the diagram to a file
            diagram_path = self.output_dir / f"{repo_name}_class_diagram.md"
            with open(diagram_path, "w", encoding="utf-8") as f:
                f.write(diagram_code)
 
            return str(diagram_path)
       
        except Exception as e:
            logging.error(f"Error generating UML class diagram: {e}")
            return f"Error generating UML class diagram: {str(e)}"
 
    def _generate_flow_diagram(self, contents: Dict[str, dict], repo_name: str) -> str:
        """
        Generate flow diagrams for legacy languages (C, COBOL) and .NET languages.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            repo_name (str): Name of the repository.
            
        Returns:
            str: Path to the generated flow diagram in Mermaid format.
        """
        if not self.config.include_flow_diagram:
            return None
        try:
            # Define supported languages for flow diagram generation
            supported_languages = ["C", "COBOL", "C#", "VB.NET", "F#", "ASP.NET",
                "JavaScript", "TypeScript", "Next.js", "React", "RPG", "RPG ILE", "SQL RPG", "CL Program", "CL ILE", "DDS Source", "Display File", "Printer File", "Logical File", "Physical File"]
            
            # Collect code samples from legacy languages and .NET languages
            target_files = []
            for path, info in contents.items():
                if info["type"] == "file" and info["language"] in supported_languages:
                    target_files.append((path, info))
            
            # Take a representative sample of files
            sample_files = target_files[:min(10, len(target_files))]
            
            # Extract code snippets for analysis
            code_samples = []
            for path, info in sample_files:
                code_samples.append({
                    "path": path,
                    "content": info["content"][:8000],  # Limit content length
                    "language": info["language"]
                })
            
            # Determine the primary language type for context
            language_counts = {}
            for _, info in sample_files:
                lang = info["language"]
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            primary_language = max(language_counts, key=language_counts.get) if language_counts else "Mixed"
            
            # Check if we have .NET languages
            dotnet_languages = {"C#", "VB.NET", "F#", "ASP.NET"}
            has_dotnet = any(lang in dotnet_languages for lang in language_counts.keys())
            
            # Generate flow diagram extraction prompt with .NET-specific context
            if has_dotnet:
                prompt = f"""
                Analyze the following code samples from a system containing .NET and legacy code and generate a flow diagram in Mermaid syntax:
                
                Primary Language: {primary_language}
                Code Samples: {code_samples}
                
                Rules for the diagram:
                1. Focus on the main program flow and key methods/functions/subroutines
                2. Show control flow between classes, methods, and modules
                3. For .NET code, highlight:
                - Class interactions and inheritance
                - Method calls and dependencies
                - Event handling and delegates (if present)
                - Controller actions (for ASP.NET)
                - Service layer interactions
                4. For legacy code (C/COBOL), highlight:
                - Main program flow and key functions/subroutines
                - Control flow between modules/functions
                5. Show key decision points, loops, and conditional flows
                6. Use proper Mermaid flowchart syntax
                7. Keep the diagram clear and readable
                8. Group related components when possible
                9. Do not include any round brackets () in the diagram
                10. Do not include any double quotes (") in the diagram
                11. Do not include any single quotes (') in the diagram
                12. Use meaningful node names that represent the actual code structure
                """
            else:
                prompt = f"""
                Analyze the following code samples from a legacy system (C/COBOL) and generate a flow diagram in Mermaid syntax:
                
                {code_samples}
                
                Rules for the diagram:
                1. Focus on the main program flow and key functions/subroutines
                2. Show control flow between modules/functions
                3. Highlight key decision points and loops
                4. Use proper Mermaid flowchart syntax
                5. Keep the diagram clear and readable
                6. Do not include any round brackets () in the diagram
                7. Do not include any double quotes (") in the diagram
                8. Do not include any single quotes (') in the diagram
                """
            
            # Use GPT-4 to generate the flow diagram
            system_message = "You are an expert in analyzing code and generating flow diagrams in Mermaid syntax."
            if has_dotnet:
                system_message += " You have extensive knowledge of .NET architecture patterns, object-oriented design, and modern software development practices."
            
            diagram_response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            
            mermaid_diagram = diagram_response["choices"][0]["message"]["content"]
            
            # Extract the diagram code from the response
            diagram_code_match = re.search(r"```mermaid\n(.*?)```", mermaid_diagram, re.DOTALL)
            diagram_code = diagram_code_match.group(1) if diagram_code_match else mermaid_diagram
            
            # Remove or replace unwanted characters (round brackets, double quotes, and single quotes)
            diagram_code = diagram_code.replace("(", "").replace(")", "")  # Remove round brackets
            diagram_code = diagram_code.replace('"', "")  # Remove double quotes
            diagram_code = diagram_code.replace("'", "")  # Remove single quotes
            
            
            # Save the diagram to a file
            diagram_path = self.output_dir / f"{repo_name}_flow_diagram.md"
            with open(diagram_path, "w", encoding="utf-8") as f:
                f.write(diagram_code)

            
            return str(diagram_path)
            
        except Exception as e:
            logging.error(f"Error generating flow diagram: {e}")
            return f"Error generating flow diagram: {str(e)}"

    def _generate_er_diagram(self, contents: Dict[str, dict], repo_name: str) -> str:
        """
        Generate ER diagrams for SQL-based languages (SQL, PL/SQL, MySQL).
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            repo_name (str): Name of the repository.
            
        Returns:
            str: Path to the generated ER diagram in Mermaid format.
        """
        if not self.config.include_er_diagram:
            return None
        try:
            # Collect SQL files
            sql_files = []
            for path, info in contents.items():
                if info["type"] == "file" and info["language"] in ["SQL", "PL/SQL", "MySQL"]:
                    sql_files.append((path, info))
            
            # Take a representative sample of files
            sample_files = sql_files[:min(10, len(sql_files))]
            
            # Extract SQL snippets for analysis
            sql_samples = []
            for path, info in sample_files:
                sql_samples.append({
                    "path": path,
                    "content": info["content"][:8000]  # Limit content length
                })
            
            # Generate ER diagram extraction prompt
            prompt = f"""
            Analyze the following SQL code samples and generate an ER diagram in Mermaid syntax:
            
            {sql_samples}
            
            Rules for the diagram:
            1. Identify tables and their relationships
            2. Show primary keys, foreign keys, and cardinality
            3. Include important columns and their data types
            4. Use proper Mermaid ER diagram syntax
            5. Keep the diagram clear and readable
            """
            
            # Use GPT-4 to generate the ER diagram
            diagram_response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in database design and generating ER diagrams in Mermaid syntax."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            
            mermaid_diagram = diagram_response["choices"][0]["message"]["content"]
            
            # Extract the diagram code from the response
            diagram_code_match = re.search(r"```mermaid\n(.*?)```", mermaid_diagram, re.DOTALL)
            diagram_code = diagram_code_match.group(1) if diagram_code_match else mermaid_diagram
            
            # Save the diagram to a file
            diagram_path = self.output_dir / f"{repo_name}_er_diagram.md"
            with open(diagram_path, "w", encoding="utf-8") as f:
                f.write("\n\n")
                f.write(diagram_code)

            return str(diagram_path)
            
        except Exception as e:
            logging.error(f"Error generating ER diagram: {e}")
            return f"Error generating ER diagram: {str(e)}"

    def generate_reference_architecture_diagram(self, contents: Dict[str, dict], repo_name: str) -> str:
        """
        Generate an enhanced reference architecture diagram for OOPS languages based on the repository's structure.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents.
            repo_name (str): Name of the repository.
            
        Returns:
            str: Path to the generated reference architecture diagram in Mermaid format.
        """
        if not self.config.include_reference_architecture:
            return None
        try:
            # Determine the primary language of the repository
            primary_language = self._determine_primary_language(contents)
            
            supported_languages = [
                "Java", "C++", "C#", "Python", "JavaScript", 
                "TypeScript", "VB.NET", "ASP.NET Razor", 
                "HTML", "CSS", "Visual Basic .NET", 
                ".NET Assembly", "XML", "ABAP","COBOL","SQL","C Header",
                "RPG", "RPG ILE", "SQL RPG", "CL Program", "CL ILE", "DDS Source", "Display File", "Printer File", "Logical File", "Physical File"
            ]
            
            if primary_language not in supported_languages:
                logging.warning(f"Unsupported primary language for reference architecture diagram: {primary_language}")
                return None

            # Collect code samples from supported languages
            relevant_files = []
            for path, info in contents.items():
                if info["type"] == "file" and info["language"] in supported_languages:
                    relevant_files.append((path, info))
            
            # Take a representative sample of files - prioritize important files
            # Look for key architecture files first
            key_patterns = [
                r"config", r"service", r"controller", r"model", 
                r"repository", r"dao", r"api", r"interface",
                r"impl", r"factory", r"manager", r"util", r"helper",
                r"view", r"component", r"entity", r"dto", r"domain"
            ]
            
            # Categorize files by architectural layer
            layer_patterns = {
                "presentation": [r"view", r"component", r"ui", r"page", r"screen", r"form", r"template"],
                "controller": [r"controller", r"api", r"endpoint", r"resource", r"route"],
                "service": [r"service", r"manager", r"handler", r"processor", r"orchestrator"],
                "domain": [r"model", r"entity", r"domain", r"vo", r"dto", r"pojo", r"bean"],
                "persistence": [r"repository", r"dao", r"mapper", r"store", r"persistence"],
                "infrastructure": [r"config", r"util", r"helper", r"factory", r"provider", r"client", r"connector"]
            }
            
            # Categorize files by layer
            layer_files = {layer: [] for layer in layer_patterns}
            
            for path, info in relevant_files:
                for layer, patterns in layer_patterns.items():
                    if any(re.search(pattern, path, re.IGNORECASE) for pattern in patterns):
                        layer_files[layer].append((path, info))
                        break
            
            # Ensure we have a balanced representation of all layers
            key_files = []
            for layer, files in layer_files.items():
                # Take up to 3 files from each layer
                key_files.extend(files[:min(3, len(files))])
            
            # Add any additional key files that might have been missed
            for path, info in relevant_files:
                if (path, info) not in key_files and any(re.search(pattern, path, re.IGNORECASE) for pattern in key_patterns):
                    key_files.append((path, info))
                    if len(key_files) >= 15:  # Cap at 15 files total
                        break
            
            # Extract code snippets for analysis
            code_samples = []
            for path, info in key_files:
                code_samples.append({
                    "path": path,
                    "language": info["language"],
                    "content": info["content"][:5000]  # Limit content length for more focused analysis
                })
            
            # Generate enhanced Mermaid diagram extraction prompt
            prompt = f"""
            Analyze the following code samples from a {primary_language} repository and generate a comprehensive reference architecture diagram in Mermaid syntax:
            
            {json.dumps(code_samples, indent=2)}
            
             Rules for creating an effective Mermaid diagram:
            # Comprehensive Mermaid Architectural Diagram Specification
 
                === CORE REQUIREMENTS ===
                1. STRUCTURE:
                • Type: flowchart TD (Top-Down)
                • Title: "{repo_name} Reference Architecture"
                • Mandatory Layers (vertical order):
                    1. Presentation/UI (top)
                    2. API/Controllers
                    3. Services
                    4. Domain Models
                    5. Data Access
                    6. Database (bottom)
                    7. Infrastructure (right-aligned)
                
                2. COMPONENTS:
                • Minimum 2 components per layer
                • Include all implied components
                • Clear layer boundaries
                • Highlight cross-cutting concerns
                
                === VISUAL DESIGN ===
                3. ICON SCHEME:
                ⎔ UI:        fa:fa-window-maximize
                ⇄ API:       fa:fa-exchange
                ⚙ Service:   fa:fa-cogs
                ▣ Domain:    fa:fa-cubes
                🗄 DAO:       fa:fa-database
                🖥 Database:  fa:fa-server
                ⛨ Security:  fa:fa-shield
                ⚒ Config:    fa:fa-wrench
                ☁ Cloud:     fa:fa-cloud
                🔌 External:  fa:fa-plug
                
                4. COLOR SCHEME:
                🟢 UI:        #d4ffcc (light green)
                🟠 API:       #ffe0cc (light orange)
                🔵 Service:   #cce5ff (light blue)
                🟡 Domain:    #ffffcc (light yellow)
                🟣 DAO:       #e6ccff (light purple)
                🎀 DB:        #ffd6e0 (pink)
                ⚪ Infra:     #e6e6e6 (light gray)
                
                === SYNTAX SPECIFICATION ===
                5. NODE FORMAT:
                • Pattern: ComponentType_Name[Label fa:fa-icon]
                • Example: Service_Auth[AuthService fa:fa-cogs]
                • Forbidden: ()"' spaces in IDs
                
                6. RELATIONSHIPS:
                →  -->  : Dependency
                ⬭ --o  : Aggregation
                ⤳ -.-> : Cross-cutting
                ◈ --*  : Composition
                ▷ --|> : Inheritance
                
                7. SUBGRAPH TEMPLATE:
                subgraph Layer_Name [fa:fa-icon]
                    Node1[Label fa:fa-icon]
                    Node2[Label fa:fa-icon]
                end
                
                === VALIDATION CRITERIA ===
                8. CONNECTION RULES:
                • UI → Controllers ONLY
                • Controllers → Services ONLY
                • Services → DAO + Domain
                • DAO → Database ONLY
                • Infra -.-> Any Layer
                
                9. MANDATORY ELEMENTS:
                ✓ Security configuration
                ✓ Database access chain
                ✓ ≥3 cross-layer flows
                ✓ Infrastructure links
                ✓ All layers represented
                
                === ERROR PREVENTION ===
                10. CRITICAL CHECKS:
                    [ ] No spaces after commas in class lists
                    [ ] All icons match layer type
                    [ ] No orphaned nodes
                    [ ] Valid Mermaid syntax
                    [ ] Color consistency
                
                11. PROHIBITED:
                    × Mixed layer connections
                    × Missing DB termination
                    × Unstyled components
                    × Special chars in IDs
                
                === EXAMPLE IMPLEMENTATION ===
                flowchart TD
                    subgraph Presentation [fa:fa-window-maximize]
                        UI_Dash[Dashboard fa:fa-window-maximize]
                    end
                
                    subgraph Controllers [fa:fa-exchange]
                        Ctrl_User[UserAPI fa:fa-exchange]
                    end
                
                    subgraph Infrastructure [fa:fa-shield]
                        Sec_JWT[JWT Config fa:fa-shield]
                    end
                
                    UI_Dash --> Ctrl_User
                    Sec_JWT -.-> Ctrl_User
                    class UI_Dash,UI_Login presentation
                    class Ctrl_User controller
                
                === SPECIAL COMPONENTS ===
                12. EDGE CASES:
                    • Microservices: ▣▣ Double cubes
                    • Legacy: ⚠ with #ffcccc
                    • Queues: 🗄→🗄 with dashed line
                    • Caches: 🗄 with glow effect
                
                13. LAYER BOUNDARIES:
                    • Clear vertical separation
                    • Infrastructure on right
                    • Database as terminus
                    • No layer skipping
                
            """
            
            # Use GPT-4 to generate the reference architecture diagram
            diagram_response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in analyzing code structure and generating reference architecture diagrams in Mermaid syntax.
                        You understand software design patterns and can visualize complex architectures clearly.
                        You always include ALL standard architectural layers in your diagrams regardless of whether they appear in the code samples.
                        Your diagrams use appropriate icons, arrow formats, and styling to enhance readability and represent the complete system architecture."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1  # Lower temperature for more consistent results
            )
            
            mermaid_diagram = diagram_response["choices"][0]["message"]["content"]
            
            # Extract the diagram code from the response
            diagram_code_match = re.search(r"```mermaid\n(.*?)```", mermaid_diagram, re.DOTALL)
            diagram_code = diagram_code_match.group(1) if diagram_code_match else mermaid_diagram
            
            # Process the diagram code: remove forbidden characters and ensure proper formatting
            diagram_code = diagram_code.replace("(", "_").replace(")", "_")  # Replace brackets with underscores
            diagram_code = diagram_code.replace('"', "").replace("'", "")  # Remove quotes
            
            # Add comprehensive styling directives for better visualization
            style_directives = """
    %% Style definitions for all architectural layers
    classDef presentationLayer fill:#d4ffcc,stroke:#3b3,stroke-width:1px
    classDef controllerLayer fill:#ffe0cc,stroke:#f95,stroke-width:1px
    classDef serviceLayer fill:#cce5ff,stroke:#36f,stroke-width:1px
    classDef domainLayer fill:#ffffcc,stroke:#cc0,stroke-width:1px
    classDef dataAccessLayer fill:#e6ccff,stroke:#93f,stroke-width:1px
    classDef infrastructureLayer fill:#e6e6e6,stroke:#666,stroke-width:1px
    classDef databaseLayer fill:#ffd6e0,stroke:#f69,stroke-width:1px
    classDef securityLayer fill:#ffcccc,stroke:#f66,stroke-width:1px

    %% Apply styles to specific component types
    class UI,View,Component,Page,Screen,Form,Template presentationLayer
    class Controller,API,Endpoint,Resource,Route controllerLayer
    class Service,Manager,Handler,Processor,Orchestrator serviceLayer
    class Model,Entity,Domain,DTO,POJO,Bean,VO domainLayer
    class Repository,DAO,Mapper,Store dataAccessLayer
    class Config,Util,Helper,Factory,Provider infrastructureLayer
    class Database,Storage,Cache databaseLayer
    class Security,Auth,Authentication,Authorization securityLayer
            """
            
            # Add the style directives after the diagram type declaration
            if "flowchart" in diagram_code:
                lines = diagram_code.split('\n')
                for i, line in enumerate(lines):
                    if "flowchart" in line:
                        lines.insert(i + 1, style_directives)
                        break
                diagram_code = '\n'.join(lines)
            else:
                diagram_code = f"flowchart TD\n{style_directives}\n{diagram_code}"
            
            # Save the diagram to a file
            diagram_path = self.output_dir / f"{repo_name}_reference_architecture_diagram.md"
            with open(diagram_path, "w", encoding="utf-8") as f:
                
                f.write(diagram_code)
             

            return str(diagram_path)
            
        except Exception as e:
            logging.error(f"Error generating reference architecture diagram: {e}")
            return f"Error generating reference architecture diagram: {str(e)}"

    def _create_class_extraction_prompt(self, language: str, code_samples: List[Dict[str, str]]) -> str:
        """
        Create a language-specific prompt for extracting class information.
        
        Args:
            language (str): The programming language to analyze.
            code_samples (List[Dict[str, str]]): Code samples to analyze.
            
        Returns:
            str: Prompt for class extraction.
        """
        base_prompt = f"""
        Extract class information from the following {language} code samples:
        
        """
        
        # Add code samples
        for i, sample in enumerate(code_samples):
            base_prompt += f"FILE {i+1}: {sample['path']}\n"
            base_prompt += f"```{language.lower()}\n{sample['content']}\n```\n\n"
        
        # Add language-specific instructions
        if language == "Java":
            base_prompt += """
            For each Java class or interface found, extract:
            1. Class/interface name
            2. Access modifiers (public, private, protected)
            3. Whether it's a class, interface, abstract class, or enum
            4. Fields with their types and access modifiers
            5. Methods with their return types, parameters, and access modifiers
            6. Inheritance relationships (extends)
            7. Implementation relationships (implements)
            8. Package information
            """
        elif language == "C++":
            base_prompt += """
            For each C++ class or struct found, extract:
            1. Class/struct name
            2. Access specifiers (public, private, protected)
            3. Member variables with their types and access specifiers
            4. Methods with their return types, parameters, and access specifiers
            5. Inheritance relationships
            6. Whether methods are virtual, pure virtual, static, const
            7. Templates and template parameters
            8. Namespaces
            """
        elif language == "C":
            base_prompt += """
            For C code, extract:
            1. Structs with their names and fields
            2. Typedef structs
            3. Function prototypes that operate on these structs
            4. Relationships between structs (where one struct contains another)
            5. Enums and their values
            """
        elif language == "Python":
            base_prompt += """
            For each Python class found, extract:
            1. Class name
            2. Parent classes (inheritance)
            3. Instance variables and class variables
            4. Methods with their parameters
            5. Static methods, class methods, and properties
            6. Decorators used
            7. Relationships between classes
            8. Module information
            """
        elif language == ".NET Framework":
            base_prompt += """
            For each .NET class found, extract:
            1. Class/interface/struct name
            2. Access modifiers (public, private, protected, internal)
            3. Whether it's a class, interface, abstract class, or struct
            4. Properties with their types and access modifiers
            5. Fields with their types and access modifiers
            6. Methods with their return types, parameters, and access modifiers
            7. Inheritance relationships
            8. Interface implementations
            9. Namespace information
            10. Attributes (if any)
            """
        
        elif language == "VB.NET Framework":
            base_prompt += """
            For each VB.NET class found, extract:
            1. Class/Interface/Struct Name: The name of the class, interface, or structure.
            2. Access Modifiers: Public, Private, Protected, Friend, Protected Friend.
            3. Type: Whether it's a Class, Interface, Abstract Class (MustInherit), or Struct (Structure).
            4. Properties: Properties with their types and access modifiers.
            5. Fields: Fields with their types and access modifiers.
            6. Methods: Methods with their return types, parameters, and access modifiers.
            7. Inheritance Relationships: Base classes and derived classes.
            8. Interface Implementations: Interfaces implemented by the class.
            9. Namespace Information: Namespace where the class or interface is defined.
            10. Attributes: Any attributes applied to the class, interface, or members.
 
            """
       
        elif language == "SQL database schema":
            base_prompt +="""
            For each SQL database schema/object found, extract:
            1. Object Name: The name of the table, view, stored procedure, function, or trigger.
            2. Object Type: Whether it's a Table, View, Stored Procedure, Function, Trigger, etc.
            3. Column Information (for tables/views): Column names, data types, nullability, default values.
            4. Constraints: Primary keys, foreign keys, unique constraints, check constraints.
            5. Indexes: Index names, columns included, uniqueness, clustered/non-clustered status.
            6. Relationships: Foreign key relationships, parent-child tables.
            7. Permissions: Access control and user permissions on the object.
            8. Table Properties: Storage information, partitioning, and other table-specific attributes.
            9. Schema Information: Database schema where the object is defined.
            10. Dependencies: Objects that depend on or are referenced by this object.
           
            """
           
        elif language == "PL/SQL database schema":
            base_prompt +="""
            For each PL/SQL object found, extract:
            1. Object Name: The name of the procedure, function, package, trigger, or type.
            2. Object Type: Whether it's a Procedure, Function, Package, Package Body, Trigger, or Type.
            3. Parameters: Parameter names, data types, modes (IN, OUT, IN OUT).
            4. Return Type: The return data type for functions.
            5. Variables: Local variables with their data types and default values.
            6. Logic Structure: Main sections of code and control flow.
            7. Exception Handling: Exception handlers and error management approaches.
            8. Dependencies: Objects called or referenced within the code.
            9. Schema Information: Database schema where the object is defined.
            10. Performance Considerations: Hints, bulk operations, or optimization techniques used.
           
            """
           
        elif language == "MySQL database schema":
            base_prompt +="""
            For each MySQL database object found, extract:
            1. Object Name: The name of the table, view, stored procedure, function, or trigger.
            2. Object Type: Whether it's a Table, View, Stored Procedure, Function, Trigger, or Event.
            3. Column Information: Column names, data types, nullability, auto-increment status.
            4. Storage Engine: InnoDB, MyISAM, or other engine being used.
            5. Constraints: Primary keys, foreign keys, unique constraints.
            6. Indexes: Index names, columns included, uniqueness, type (BTREE, HASH, etc.).
            7. Triggers: Before/After Insert/Update/Delete triggers associated with tables.
            8. Partitioning: Any partitioning scheme applied to tables.
            9. Character Set and Collation: Table and column-level character sets.
            10. Database Information: Database where the object is defined.
           
            """
        
         # Add language-specific instructions
        elif language == "Visual Basic .NET":
            base_prompt += """
            For each VB.NET class found, extract:
            1. Class/Interface/Struct Name: The name of the class, interface, or structure.
            2. Access Modifiers: Public, Private, Protected, Friend, Protected Friend.
            3. Type: Whether it's a Class, Interface, Abstract Class (MustInherit), or Struct (Structure).
            4. Properties: Properties with their types and access modifiers.
            5. Fields: Fields with their types and access modifiers.
            6. Methods: Methods with their return types, parameters, and access modifiers.
            7. Inheritance Relationships: Base classes and derived classes.
            8. Interface Implementations: Interfaces implemented by the class.
            9. Namespace Information: Namespace where the class or interface is defined.
            10. Attributes: Any attributes applied to the class, interface, or members.
            """
        
        elif   language == "SAP ABAP":
            base_prompt += """
            For each SAP ABAP object found, extract:
            1. Object Name: The name of the class, program, function module, or other object.
            2. Object Type: Whether it's a Class, Interface, Program, Function Group, Function Module, Table, Data Element, Domain, Structure, or Report.
            3. Package Information: The package/development class the object belongs to.
            4. Inheritance Hierarchy: For classes, identify parent classes and implemented interfaces.
            5. Visibility Sections: Public, Protected, Private sections in classes and the members defined in each.
            6. Method Definitions: Method names, parameters (importing, exporting, changing, returning), exceptions raised.
            7. Data Definitions: Global and local variable definitions with their types and technical attributes.
            8. Database Operations: Any SELECT, INSERT, UPDATE, DELETE operations and the tables they interact with.
            9. RFC Calls: Any remote function calls to other systems.
            10. BAPI Usage: Any Business API calls and their purposes.
            11. Authorization Checks: Any authority-check statements and the authorization objects they use.
            12. Message Classes: References to message classes and the messages being used.
            13. Transaction Codes: Any transaction codes referenced or defined.
            14. Enhancement Points: Any enhancement points or BAdIs implemented or used.
            15. Documentation: Comments and documentation strings for the objects.
            """
        
        base_prompt += """
        Format the output as a structured overview of each class with its properties, methods, and relationships.
        Focus on the structure rather than implementation details.
        """
        
        return base_prompt
    
    def _get_language_specific_config(self, language: str) -> Dict[str, any]:
        """
        Get language-specific configuration for analysis and documentation.
        """
        configs = {
            # Default configuration
            "default": {
                "analysis_depth": "moderate",
                "focus_areas": ["structure", "functions", "dependencies"],
                "diagram_type": "class",
                "complexity_metrics": ["cyclomatic", "inheritance"],
                "key_file_patterns": [
                    r"main\.", r"app\.", r"core\.", r"service\.", 
                    r"controller\.", r"model\.", r"repository\."
                ]
            },
            
            # COBOL configuration (existing)
            "COBOL": {
                "analysis_depth": "detailed",
                "focus_areas": ["divisions", "paragraphs", "files", "tables"],
                "diagram_type": "flowchart",
                "complexity_metrics": ["paragraph_depth", "perform_nesting"],
                "key_file_patterns": [
                    r"MAIN\.", r"DRIVER\.", r"CONTROL\.", 
                    r"FILE[0-9]*\.", r"REPORT[0-9]*\."
                ],
                "special_analysis": {
                    "division_analysis": True,
                    "copybook_analysis": True,
                    "file_definition_analysis": True
                }
            },
            
            # Java configuration (existing)
            "Java": {
                "analysis_depth": "detailed",
                "focus_areas": ["classes", "interfaces", "packages", "dependencies"],
                "diagram_type": "class",
                "complexity_metrics": ["inheritance_depth", "coupling"],
                "key_file_patterns": [
                    r"Main\.java", r"Application\.java", r"Controller\.java",
                    r"Service\.java", r"Repository\.java", r"Model\.java"
                ],
                "special_analysis": {
                    "spring_analysis": True,
                    "jpa_analysis": True,
                    "design_patterns": True
                }
            },
            
            # VB.NET configuration (existing)
            "VB.NET": {
                "analysis_depth": "moderate",
                "focus_areas": ["forms", "modules", "classes", "controls"],
                "diagram_type": "class",
                "complexity_metrics": ["inheritance_depth", "form_complexity"],
                "key_file_patterns": [
                    r"Form[0-9]*\.vb", r"Module[0-9]*\.vb", 
                    r"Class[0-9]*\.vb", r"Service\.vb"
                ],
                "special_analysis": {
                    "winforms_analysis": True,
                    "legacy_component_analysis": True
                }
            },
            
            # SQL configuration (existing)
            "SQL": {
                "analysis_depth": "detailed",
                "focus_areas": ["tables", "procedures", "functions", "triggers"],
                "diagram_type": "er",
                "complexity_metrics": ["join_complexity", "nested_queries"],
                "key_file_patterns": [
                    r"CREATE_.*\.sql", r"PROC_.*\.sql", 
                    r"FUNC_.*\.sql", r"VIEW_.*\.sql"
                ],
                "special_analysis": {
                    "table_relationship_analysis": True,
                    "stored_procedure_analysis": True,
                    "trigger_analysis": True
                }
            },
            
            # C configuration (existing)
            "C": {
                "analysis_depth": "moderate",
                "focus_areas": ["functions", "structs", "headers", "macros"],
                "diagram_type": "flowchart",
                "complexity_metrics": ["function_depth", "pointer_usage"],
                "key_file_patterns": [
                    r"main\.c", r".*\.h", r"lib.*\.c", r"util.*\.c"
                ],
                "special_analysis": {
                    "header_dependency_analysis": True,
                    "memory_management_analysis": True
                }
            },
            
            # ABAP configuration (existing)
            "ABAP": {
                "analysis_depth": "detailed",
                "focus_areas": ["reports", "function_modules", "bapis", "tables"],
                "diagram_type": "flowchart",
                "complexity_metrics": ["module_complexity", "authorization_checks"],
                "key_file_patterns": [
                    r"Z.*\.abap", r"Y[A-Z0-9_]+\.", r"^R[A-Z0-9_]+\.", 
                    r"^SAPM[A-Z0-9_]+\.", r"^L[A-Z0-9_]+\.", r"^F[A-Z0-9_]+"
                ],
                "special_analysis": {
                    "sap_module_analysis": True,
                    "bapi_analysis": True,
                    "authorization_check_analysis": True
                }
            },
            
            # Python configuration (NEW)
            "Python": {
                "analysis_depth": "detailed",
                "focus_areas": ["classes", "functions", "modules", "packages", "dependencies"],
                "diagram_type": "class",
                "complexity_metrics": ["cyclomatic_complexity", "inheritance_depth", "import_coupling"],
                "key_file_patterns": [
                    r"main\.py", r"app\.py", r"__init__\.py", r"settings\.py",
                    r"models\.py", r"views\.py", r"urls\.py", r"forms\.py",
                    r"admin\.py", r"serializers\.py", r"tests\.py"
                ],
                "special_analysis": {
                    "django_analysis": True,
                    "flask_analysis": True,
                    "fastapi_analysis": True,
                    "async_analysis": True,
                    "decorator_analysis": True,
                    "exception_handling_analysis": True,
                    "data_structure_analysis": True
                }
            },
            
            # RPG configuration (existing - already implemented)
            "RPG": {
                "analysis_depth": "detailed",
                "focus_areas": ["procedures", "subprocedures", "files", "indicators"],
                "diagram_type": "flowchart",
                "complexity_metrics": ["subroutine_depth", "file_operations"],
                "key_file_patterns": [
                    r"MAIN\.", r"DRIVER\.", r"CONTROL\.", r"PGM[0-9]*\."
                ],
                "special_analysis": {
                    "cycle_analysis": True,
                    "file_definition_analysis": True,
                    "copybook_analysis": True
                }
            },
            
            # DDS configuration (NEW)
            "DDS Source": {
                "analysis_depth": "detailed",
                "focus_areas": ["file_specifications", "record_formats", "field_definitions", "keywords"],
                "diagram_type": "flowchart",
                "complexity_metrics": ["field_complexity", "record_relationships"],
                "key_file_patterns": [
                    r".*DSPF", r".*PRTF", r".*LF", r".*PF", r".*DDS"
                ],
                "special_analysis": {
                    "file_type_analysis": True,
                    "field_definition_analysis": True,
                    "keyword_analysis": True,
                    "record_format_analysis": True
                }
            }
        }
        
        config_path = os.path.join(self.output_dir, '..', 'instance', 'language_configs.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_configs = json.load(f)
                    if language in custom_configs:
                        return custom_configs[language]
            except Exception as e:
                logging.error(f"Error loading custom language configs: {e}")
        
        return configs.get(language, configs["default"])
    
    def _analyze_with_language_config(self, content: str, language: str, file_path: str) -> str:
        """
        Analyze code with language-specific configuration.
        """
        lang_config = self._get_language_specific_config(language)
        prompt = self._build_language_specific_prompt(content, language, file_path, lang_config)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a senior {language} architect. Provide concise, actionable code analysis."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                    request_timeout=30
                )
                return response["choices"][0]["message"]["content"]
                
            except openai.error.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait_time = (attempt + 1) * 5
                logging.warning(f"Rate limited, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
            except Exception as e:
                logging.error(f"Error analyzing {file_path}: {str(e)}")
                return f"Error analyzing file: {str(e)}"


    def generate_documentation_from_contents(self, contents: Dict[str, dict], repo_name: str) -> Dict[str, str]:
        """
        Generate documentation from directly provided contents instead of fetching from GitHub.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents
            repo_name (str): Name to use for the repository
            
        Returns:
            Dict[str, str]: Dictionary of generated documentation paths
        """
        try:
            # Generate directory tree if needed
            directory_tree = ""
            if self.config.include_code_structure:
                directory_tree = self._generate_directory_tree(contents)

            # Identify key file types
            language_counts = {}
            for path, info in contents.items():
                if info["type"] == "file" and "language" in info:
                    lang = info["language"]
                    language_counts[lang] = language_counts.get(lang, 0) + 1

            # Sort languages by frequency
            sorted_languages = sorted(
                language_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_languages = [lang for lang, count in sorted_languages[:5]]

            # Analyze top files if needed
            file_analyses = {}
            if self.config.max_files_to_analyze > 0:
                key_files = self.select_key_files(contents, self.config.max_files_to_analyze)
                file_analyses = self.analyze_files_parallel(key_files[:self.config.max_files_to_analyze])

            # Get code quality analysis if needed
            line_of_code = {}
            loc_chart_path = None
            if self.config.include_loc_analysis or self.config.include_loc_chart:
                line_of_code = self.analyze_code_quality(contents)
                if self.config.include_loc_chart:
                    loc_chart_path = self.create_lines_of_code_chart(
                        line_of_code["language_loc"], repo_name
                    )

            # Generate diagrams if needed
            diagrams = {}
            if any([
                self.config.include_class_diagram,
                self.config.include_flow_diagram,
                self.config.include_er_diagram,
                self.config.include_reference_architecture
            ]):
                diagrams = self.generate_diagrams_parallel(contents, repo_name)
                # Filter out diagrams that weren't requested
                if not self.config.include_class_diagram:
                    diagrams.pop("class_diagram", None)
                if not self.config.include_flow_diagram:
                    diagrams.pop("flow_diagram", None)
                if not self.config.include_er_diagram:
                    diagrams.pop("er_diagram", None)
                if not self.config.include_reference_architecture:
                    diagrams.pop("reference_architecture", None)

            # Get code complexity analysis if needed
            complexity_metrics = {}
            complexity_charts = {}
            if self.config.include_complexity_analysis or self.config.include_complexity_charts:
                complexity_metrics = self.analyze_code_complexity(contents)
                if self.config.include_complexity_charts:
                    complexity_charts = self.create_complexity_charts(complexity_metrics, repo_name)

            # Extract SQL objects if needed
            sql_objects = {}
            if self.config.include_sql_objects and any(lang in ["SQL", "PL/SQL", "MySQL"] for lang in language_counts):
                sql_objects = self._extract_sql_objects(contents)

            # Generate project documentation using GPT-4-32k
            prompt = self._build_documentation_prompt(
                repo_name,
                directory_tree,
                top_languages,
                file_analyses,
                line_of_code,
                complexity_metrics,
                sql_objects
            )

            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior software architect with extensive expertise in analyzing and documenting codebases across a wide range of legacy and modern technologies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            documentation = response["choices"][0]["message"]["content"]

            # Save as markdown
            md_path = self.output_dir / f"{repo_name}_DOCUMENTATION.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(documentation)

            # Generate PDF with proper code block formatting for the tree
            pdf_path = self.markdown_to_pdf(
                documentation, 
                f"{repo_name}_DOCUMENTATION.pdf", 
                loc_chart_path,
                diagrams.get("class_diagram"),
                complexity_charts,
                sql_objects,
                contents,
                repo_name
            )

            # Prepare result dictionary with all generated artifacts
            result = {
                "markdown_path": str(md_path), 
                "pdf_path": str(pdf_path),
            }
            
            # Add diagram paths explicitly to ensure they're included
            if diagrams:
                # Explicitly add important diagrams with clear keys
                if "class_diagram" in diagrams and diagrams["class_diagram"] is not None:
                    result["class_diagram_path"] = str(diagrams["class_diagram"])
                if "reference_architecture" in diagrams and diagrams["reference_architecture"] is not None:
                    result["reference_architecture_path"] = str(diagrams["reference_architecture"])
                if "flow_diagram" in diagrams and diagrams["flow_diagram"] is not None:
                    result["flow_diagram_path"] = str(diagrams["flow_diagram"])
                if "er_diagram" in diagrams and diagrams["er_diagram"] is not None:
                    result["er_diagram_path"] = str(diagrams["er_diagram"])
                
            # Add chart paths if they were generated
            if loc_chart_path:
                result["loc_chart_path"] = str(loc_chart_path)
            if complexity_charts:
                result["complexity_charts"] = {
                    k: str(v) for k, v in complexity_charts.items()
                }
            
            return result

        except Exception as e:
            logging.error(f"Error in documentation generation: {e}")
            raise
    
    def process_uploaded_zip(self, zip_file_path: str) -> Dict[str, dict]:
        """
        Process an uploaded ZIP file and extract its contents for analysis.
        Optimized for speed and better inheritance analysis.
        
        Args:
            zip_file_path (str): Path to the uploaded ZIP file
            
        Returns:
            Dict[str, dict]: Dictionary containing repository contents in the same format as fetch_repository_contents
        """
        contents = {}
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # First pass: scan and catalog all class names for inheritance analysis
            class_catalog = self._scan_classes(temp_dir)
                
            # Use multiple threads to process files
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as executor:
                futures = []
                
                # Walk through the extracted directory
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, temp_dir)
                        
                        # Skip files that match exclude patterns immediately
                        if any(fnmatch.fnmatch(rel_path, pattern) for pattern in self.exclude_patterns):
                            continue
                            
                        # Get file extension
                        ext = os.path.splitext(file)[1][1:].lower()
                        if ext not in self.supported_extensions:
                            continue
                        
                        # Submit file processing to thread pool
                        futures.append(
                            executor.submit(
                                self._process_zip_file, 
                                file_path, 
                                rel_path, 
                                ext, 
                                class_catalog
                            )
                        )
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        path, file_info = result
                        contents[path] = file_info
                        
        except Exception as e:
            logging.error(f"Error processing ZIP file: {e}")
            raise
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        return contents
    
    
    def _process_zip_file(self, file_path: str, rel_path: str, ext: str, class_catalog: Dict[str, str]) -> tuple:
        """
        Process a single file from the ZIP archive.
        
        Args:
            file_path (str): Absolute path to the file
            rel_path (str): Relative path within the extraction directory
            ext (str): File extension
            class_catalog (Dict[str, str]): Catalog of all classes for inheritance analysis
            
        Returns:
            tuple: (rel_path, file_info) or None if file should be skipped
        """
        try:
            # Skip if file is too large
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return None
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            language = self.supported_extensions[ext]
            file_info = {
                "content": content,
                "language": language,
                "size": file_size,
                "type": "file"
            }
            
            # Enhanced inheritance analysis
            inheritance_info = self._analyze_inheritance(content, language, class_catalog)
            if inheritance_info:
                file_info["inheritance"] = inheritance_info
                
            return rel_path, file_info
            
        except UnicodeDecodeError:
            # Skip binary files
            return None
        except Exception as e:
            logging.warning(f"Error processing file {rel_path}: {e}")
            return None
    
    def _analyze_inheritance(self, content: str, language: str, class_catalog: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze class inheritance relationships.
        
        Args:
            content (str): File content
            language (str): Programming language
            class_catalog (Dict[str, str]): Catalog of all classes and their locations
            
        Returns:
            Dict[str, Any]: Information about inheritance
        """
        inheritance_info = {}
        
        try:
            if language == "Java":
                # Match class definitions with inheritance in Java
                class_pattern = r'(public|private|protected|)\s+class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?'
                matches = re.finditer(class_pattern, content)
                
                for match in matches:
                    class_name = match.group(2)
                    parent_class = match.group(3)
                    interfaces = match.group(4)
                    
                    class_info = {"name": class_name}
                    if parent_class:
                        class_info["extends"] = parent_class
                        # Track inheritance path
                        if parent_class in class_catalog:
                            class_info["parent_file"] = class_catalog[parent_class]
                    
                    if interfaces:
                        class_info["implements"] = [i.strip() for i in interfaces.split(',')]
                    
                    if "extends" in class_info or "implements" in class_info:
                        inheritance_info[class_name] = class_info
                        
            elif language == "C#":
                # Match class definitions with inheritance in C#
                class_pattern = r'(public|private|protected|internal|)\s+class\s+(\w+)(?:\s*:\s*([\w,\s]+))?'
                matches = re.finditer(class_pattern, content)
                
                for match in matches:
                    class_name = match.group(2)
                    inheritance = match.group(3)
                    
                    class_info = {"name": class_name}
                    if inheritance:
                        # C# doesn't distinguish between extends and implements in the syntax
                        inheritances = [i.strip() for i in inheritance.split(',')]
                        if inheritances:
                            class_info["inherits"] = inheritances
                            # Track first inheritance as main parent
                            if inheritances[0] in class_catalog:
                                class_info["parent_file"] = class_catalog[inheritances[0]]
                    
                    if "inherits" in class_info:
                        inheritance_info[class_name] = class_info
                        
        except Exception as e:
            logging.debug(f"Error analyzing inheritance: {e}")
            
        return inheritance_info
    
    def _scan_classes(self, directory: str) -> Dict[str, str]:
        """
        Scan the directory to catalog all class names and their file paths.
        This helps in later inheritance analysis.
        
        Args:
            directory (str): Directory containing extracted files
            
        Returns:
            Dict[str, str]: Dictionary mapping class names to their file paths
        """
        class_catalog = {}
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                
                ext = os.path.splitext(file)[1][1:].lower()
                if ext not in self.supported_extensions:
                    continue
                    
                # Skip if file is too large
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > self.max_file_size:
                        continue
                        
                    # Quick scan for class definitions
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    language = self.supported_extensions[ext]
                    
                    # Extract class names based on language
                    if language in ["Java", "Kotlin"]:
                        # Match Java/Kotlin class definitions
                        class_matches = re.finditer(r'(public|private|protected|)\s+class\s+(\w+)', content)
                        for match in class_matches:
                            class_name = match.group(2)
                            class_catalog[class_name] = rel_path
                            
                    elif language in ["C#", "VB.NET"]:
                        # Match C# class definitions
                        if language == "C#":
                            class_matches = re.finditer(r'(public|private|protected|internal|)\s+class\s+(\w+)', content)
                        else:  # VB.NET
                            class_matches = re.finditer(r'(Public|Private|Protected|Friend|)\s+Class\s+(\w+)', content)
                            
                        for match in class_matches:
                            class_name = match.group(2)
                            class_catalog[class_name] = rel_path
                            
                except Exception:
                    continue
                    
        return class_catalog
    def generate_line_by_line_explanations(self, contents: Dict[str, dict], repo_name: str) -> Dict[str, str]:
        """
        Generate comprehensive line-by-line code explanations for all supported languages.
        
        Args:
            contents (Dict[str, dict]): Dictionary containing repository contents
            repo_name (str): Name of the repository
            
        Returns:
            Dict[str, str]: Dictionary mapping file paths to their line-by-line explanations
        """
        explanations = {}
        
        # Select key files for line-by-line explanation (limit to avoid overwhelming the PDF)
        key_files = self.select_key_files(contents, max_files=5)
        
        for file_path, file_info in key_files:
            if file_info["type"] != "file" or "language" not in file_info:
                continue
                
            language = file_info["language"]
            content = file_info["content"]
            
            # Generate language-specific line-by-line explanation
            explanation = self._generate_language_specific_line_explanation(
                content, language, file_path
            )
            
            if explanation:
                explanations[file_path] = explanation
        
        return explanations
    
    def _generate_language_specific_line_explanation(self, content: str, language: str, file_path: str) -> str:
        """
        Generate line-by-line explanation for a specific language.
        
        Args:
            content (str): File content
            language (str): Programming language
            file_path (str): Path to the file
            
        Returns:
            str: Line-by-line explanation
        """
        try:
            # Split content into lines and number them
            lines = content.split('\n')
            
            # Create the prompt based on language
            prompt = self._create_line_by_line_prompt(content, language, file_path)
            
            # Use GPT-4 to generate line-by-line explanation
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert {language} programmer and technical documentation specialist. Provide detailed line-by-line code explanations in clean, properly formatted text without using asterisks (*), hash symbols (#), or other markdown formatting. Use plain text with clear section headers and proper indentation only."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logging.error(f"Error generating line-by-line explanation for {file_path}: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def _create_line_by_line_prompt(self, content: str, language: str, file_path: str) -> str:
        """
        Create a language-specific prompt for line-by-line code explanation.
        
        Args:
            content (str): File content
            language (str): Programming language
            file_path (str): Path to the file
            
        Returns:
            str: Formatted prompt for line-by-line explanation
        """
        # Add line numbers to the content
        lines = content.split('\n')
        numbered_content = '\n'.join([f"{i+1:03d}.00\t{line}" for i, line in enumerate(lines[:100])])  # Limit to first 100 lines
        
        base_prompt = f"""
Explain Code Line by Line for the following {language} code from file: {file_path}

Code with line numbers:
```{language.lower()}
{numbered_content}
```

Provide detailed explanations in clean, properly formatted text without using asterisks (*) or hash symbols (#). Use plain text formatting with clear section headers and proper indentation.

{language} Program Documentation
Program Type: {language}
Purpose: Brief description of the program's main purpose
Files Used: List any files, databases, or external resources
Libraries: List any libraries or dependencies

A. Program Structure
A.1. Declaration Section
Explain variable declarations, file declarations, etc.

A.2. Parameters
Explain input/output parameters

A.3. File Declarations
Explain file declarations and their purposes

B. Program Logic Flow
B.1. Main Processing Loop
Explain the main program flow

B.2. Other sections as needed

C. Explain Code Line by Line
"""
        
        # Add language-specific instructions
        if language in ["CL Program", "CL ILE", "CLP"]:
            base_prompt += """
For each line, explain:
- Command purpose and syntax
- Parameter meanings and values
- File operations and their effects
- Variable assignments and manipulations
- Control flow statements (IF, DO, GOTO)
- Error handling (MONMSG)
- System interactions

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of what this line does, why it's needed, and how it fits into the overall program flow
"""
        elif language in ["RPG", "RPG ILE", "SQL RPG"]:
            base_prompt += """
For each line, explain:
- Specification type (H, F, D, I, C, O, P)
- Field definitions and data types
- File operations (READ, WRITE, UPDATE, DELETE)
- Calculation operations and built-in functions
- Indicator usage and logic
- Subprocedure calls and parameters
- SQL operations (if SQL RPG)

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the specification, operation, or calculation being performed
"""
        elif language == "COBOL":
            base_prompt += """
For each line, explain:
- Division and section purposes
- Data definitions and picture clauses
- File definitions and record structures
- Procedure division logic
- PERFORM statements and paragraph calls
- Conditional statements and loops
- File I/O operations

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the COBOL statement, data definition, or procedure
"""
        elif language == "SQL":
            base_prompt += """
For each line, explain:
- SQL statement type and purpose
- Table and column references
- JOIN conditions and relationships
- WHERE clause conditions
- Functions and expressions
- Stored procedure logic
- Transaction control statements

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the SQL statement and its database operations
"""
        elif language in ["Java", "C#", "Python", "JavaScript", "TypeScript"]:
            base_prompt += """
For each line, explain:
- Class and method declarations
- Variable declarations and initializations
- Object instantiations and method calls
- Control flow statements (if, for, while)
- Exception handling
- Import/using statements
- Business logic implementation

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the code statement, its purpose, and its role in the program
"""
        elif language in ["C", "C++"]:
            base_prompt += """
For each line, explain:
- Variable declarations and memory management
- Control flow statements
- System calls and library functions
- Struct and union definitions

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the C/C++ statement and its system-level operations
"""
        elif language == "ABAP":
            base_prompt += """
For each line, explain:
- DATA declarations and types
- SELECT statements and database operations
- LOOP and control structures
- Function module calls
- BAPI usage
- Authorization checks
- Message handling

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the ABAP statement and its SAP system interaction
"""
        elif language in ["Visual Basic .NET", "VB.NET"]:
            base_prompt += """
For each line, explain:
- Class and module declarations
- Variable declarations and initializations
- Object instantiations and method calls
- Control flow statements (If, For, While)
- Exception handling (Try-Catch)
- Imports statements
- Event handling
- Windows Forms controls

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the VB.NET statement, its purpose, and its role in the program
"""
        elif language in ["HTML", "CSS", "JavaScript", "TypeScript"]:
            base_prompt += """
For each line, explain:
- HTML tags and attributes
- CSS selectors and properties
- JavaScript/TypeScript functions and variables
- DOM manipulation
- Event handling
- API calls and async operations
- Framework-specific syntax

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the web technology statement and its browser interaction
"""
        elif language in ["Display File", "Printer File", "Logical File", "Physical File", "DDS Source"]:
            base_prompt += """
For each line, explain:
- File specification keywords
- Field definitions and attributes
- Record format specifications
- Key field definitions
- File-level keywords
- Access path specifications
- DDS keywords and their purposes

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the file specification and its database/display purpose
"""
        elif language in ["Binder Source", "Copy Member"]:
            base_prompt += """
For each line, explain:
- Copy member inclusions
- Conditional compilation directives
- Parameter definitions
- Template specifications
- System-specific configurations

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the copy member or binder specification
"""
        elif language in ["JSON", "XML", "YAML"]:
            base_prompt += """
For each line, explain:
- Data structure definitions
- Configuration parameters
- Nested object relationships
- Array and list structures
- Schema validation rules

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of the data structure and its configuration purpose
"""
        else:
            base_prompt += """
For each line, explain:
- Variable declarations and assignments
- Function/method calls
- Control flow statements
- Data operations
- System interactions
- Business logic implementation

Format each line explanation as clean text without special characters:
Line XXX.XX: Original code line
Explanation: Detailed explanation of what this line accomplishes
"""
        
        base_prompt += """

End with:
              End of Documentation                   

Provide comprehensive explanations that would help a developer understand not just what each line does, but why it's necessary and how it contributes to the overall program functionality.
"""
        
        return base_prompt

    def _build_language_specific_prompt(self, content: str, language: str, file_path: str, lang_config: Dict) -> str:
        """
        Build a language-specific analysis prompt.
        """
        common_prompt = f"""
        Analyze the {language} code file located at '{file_path}':
        
        ```{language.lower()}
        {content[:8000]}
        ```
        
        Provide concise analysis with:
        """
        
        if language == "COBOL":
            prompt = common_prompt + f"""
            1. Program purpose and divisions (IDENTIFICATION, ENVIRONMENT, DATA, PROCEDURE)
            2. Key file definitions (FD, SD) and record layouts
            3. Main paragraphs and their functions
            4. COPY book usage and dependencies
            5. Code quality assessment with focus on:
               - Paragraph complexity
               - File handling practices
               - PERFORM nesting depth
            6. Recommendations for modernization if applicable
            
            Format as bullet points with clear section headers.
            """
        elif language == "Java":
            prompt = common_prompt + f"""
            1. Class purpose and package
            2. Key methods and their responsibilities
            3. Inheritance and interface implementation
            4. Dependencies on other classes/packages
            5. Code quality assessment with focus on:
               - OO principles adherence
               - Design patterns usage
               - Coupling and cohesion
            6. Recommendations for improvement
            
            Format as bullet points with clear section headers.
            """
        elif language == "VB.NET":
            prompt = common_prompt + f"""
            1. Module/Class purpose
            2. Key methods and event handlers
            3. UI components if applicable
            4. Data access patterns
            5. Code quality assessment with focus on:
               - Event handler complexity
               - UI-business logic separation
               - Legacy code patterns
            6. Recommendations for modernization
            
            Format as bullet points with clear section headers.
            """
        elif language == "SQL":
            prompt = common_prompt + f"""
            1. Object type and purpose
            2. Key tables/columns referenced
            3. Query complexity and optimization opportunities
            4. Transaction handling
            5. Code quality assessment with focus on:
               - Query performance
               - Index usage
               - Error handling
            6. Recommendations for improvement
            
            Format as bullet points with clear section headers.
            """
        elif language == "ABAP":
            prompt = common_prompt + f"""
            1. Program/Function Module purpose
            2. Key tables and structures used
            3. BAPI/RFC calls if any
            4. Authorization checks
            5. Code quality assessment with focus on:
               - PERFORM nesting
               - Table access efficiency
               - SAP best practices
            6. Recommendations for improvement
            
            Format as bullet points with clear section headers.
            """
        elif language in ["RPG", "RPG ILE", "SQL RPG"]:
            prompt = common_prompt + f"""
            1. Program purpose and cycle type (linear/traditional)
            2. File specifications and key fields
            3. Main calculation specifications and procedures
            4. Subprocedures and prototypes
            5. Code quality assessment with focus on:
               - Indicator usage
               - File operation efficiency
               - Modern RPG practices
            6. Recommendations for modernization
            
            Format as bullet points with clear section headers.
            """
        elif language in ["CL Program", "CL ILE"]:
            prompt = common_prompt + f"""
            1. Program purpose and job control flow
            2. Key commands and parameters
            3. File and library references
            4. Error handling and monitoring
            5. Code quality assessment with focus on:
               - Command structure
               - Error handling completeness
               - System integration
            6. Recommendations for improvement
            
            Format as bullet points with clear section headers.
            """
        elif language in ["DDS Source", "Display File", "Printer File", "Logical File", "Physical File"]:
            prompt = common_prompt + f"""
            1. File type and purpose (Display, Printer, Physical, Logical)
            2. Record format definitions and field specifications
            3. Key field definitions and access paths
            4. DDS keywords and their functions
            5. Code quality assessment with focus on:
               - Field definition completeness
               - Keyword usage appropriateness
               - File relationship integrity
            6. Recommendations for modernization
            
            Format as bullet points with clear section headers.
            """
        else:
            prompt = common_prompt + f"""
            1. File purpose (1-2 sentences)
            2. Key functions/classes and their roles
            3. Notable patterns/techniques
            4. Code quality (good/fair/poor) and 2-3 specific recommendations
            
            Format as bullet points, no code examples.
            """
        
        return prompt        

