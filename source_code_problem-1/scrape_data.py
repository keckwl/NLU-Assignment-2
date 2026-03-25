import requests
from bs4 import BeautifulSoup
import time
import re
import os

SOURCES = {
    "departments": [
        "https://iitj.ac.in/computer-science-engineering/en/Faculty",
        "https://iitj.ac.in/computer-science-engineering/en/doctoral-programs",
        "https://iitj.ac.in/computer-science-engineering/en/Research-Highlights",
        "https://iitj.ac.in/civil-and-infrastructure-engineering/en/faculty-members",
        "https://iitj.ac.in/chemical-engineering/en/faculty-members",
        "https://iitj.ac.in/chemical-engineering/en/about-research",
        "https://iitj.ac.in/bioscience-bioengineering/en/Cell-Molecular-Physiology-Laboratory",
        "https://iitj.ac.in/main/en/iitj",
    ],
    "academic_programs": [
        "https://iitj.ac.in/office-of-academics/en/program-structure",
        "https://iitj.ac.in/office-of-academics/en/academic-programs",
        "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
        "https://iitj.ac.in/master-of-technology/en/eligibility",
        "https://iitj.ac.in/bachelor-of-technology/en/academic-research-facilities",
        "https://iitj.ac.in/office-of-students/en/Academics",
    ],
    "research": [
        "https://iitj.ac.in/main/en/research-highlight",
        "https://iitj.ac.in/publications/",
        "https://iitj.ac.in/main/en/research-areas-removed",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MIN_WORDS = 40

# Academic Regulations text embedded directly from the official IIT Jodhpur PDF
# Source: "Academic Programmes Rules & Regulations (For Students enrolled from July 2022 onwards)"
ACADEMIC_REGULATION_TEXT = """
Academic programmes at Indian Institute of Technology Jodhpur are designed to develop the
highest calibre human resource capable of understanding the new patterns of knowledge
creation across disciplines obliterating traditional boundaries between science humanities social
sciences and engineering. IIT Jodhpur aims to produce quality professionals who would be able
to address profound and wide-ranging societal challenges of the century such as energy food
water housing mobility and health. In addition to imparting scientific knowledge IIT Jodhpur
endeavours to inculcate human qualities of courage integrity fairness humility and team effort
among its graduates through curricular co-curricular and extra-curricular activities on campus.
The academic programmes focus on developing a temper for the life long process of learning
creative thinking and exploring. The postgraduate academic programmes focus on developing
deep understanding of the subject of study coupled with creative inquisitiveness and the ability
to address and solve new problems with free and objective mind.
The academic programmes are based on three tenets of semester system Credit System and
Relative Grading. These academic programmes are administered by a duly constituted Academic
Committee through the office of Academics. The AC will be chaired by Dean Academics. In the
absence of Dean Academics Associate Dean Academics PG or Associate Dean Academics UG
will discharge the responsibilities of Dean Academics.
The Senate of the Institute has formulated a set of guidelines and rules to ensure high standard
of performance as well as for smooth functioning of the academic programmes. Within this
broad set of guidelines subject to the approval of Senate different programmes can include
additional academic requirements as deemed necessary for that programme. These academic
programmes are continuously monitored by the Senate and make appropriate modifications
improvements as and when necessary.
The academic session normally runs from the end of July in one year to the middle of July in the
next year. The academic session is divided into three parts two regular semesters and a summer
term. Semester I runs from the fourth week of July to the last week of November. Semester II
runs from the last week of December to the last week of April. Summer Term runs from the
middle of May to the middle of July. Excluding the days of the examinations the total number
of days of instruction in a semester is at least seventy.
The academic calendar gives the exact dates of all important events during the Academic
Session such as orientation registration the commencement of classes adding and dropping of
courses submission of documents examinations submission of grades project evaluation
declaration of results mid-semester recess and vacation. This calendar is approved by the Senate.
Students are required to register for prescribed credits as per the programme on the dates
specified in the Academic Calendar. If the student does not register by the date of registration
she or he can register by paying the fine and register before the date of late registration.
A fresh student joining the Institute who is awaiting the final results of the qualifying
examination is allowed to register provisionally on submission of a certificate from her or his
last Institute stating that she or he has appeared in the final examination of the qualifying
degree. The student is required to submit documents of having passed the qualifying
examination by the last date given in the Academic Calendar for the registration to be
regularized. If the student does not register by the date of late registration as mentioned in the
Academic Calendar he or she will be deregistered. The appeal against deregistration may be
made to Chairman Senate through Dean Academics.
A student may add or drop courses with approval of the Faculty advisor till the last date of
registration. After the last date of add-drop a student will have the option of withdrawing from
a course. The last date of course withdrawal will be typically within four weeks from the
beginning of the Semester as specified in the academic calendar. The withdrawn course will be
mentioned in the Student grade card with a letter grade W.
Absence for a period of four or more weeks during a semester shall result in automatic
cancellation of the registration of a student from all the courses in that semester. A department
may offer summer courses to enable the students to clear their backlog courses and or regular
credit courses. A course will run during summer provided a faculty member is available for
running the course.
The medium of instruction is English. All courses have associated credits. Credits of a course is
based on the number of contact hours for lectures tutorials and practicals. A student on
successful completion of the course with a passing grade will earn an equivalent number of
credits. Based on the required academic maturity level of students for attending a course a
course is assigned a level. A course can consist of independent components which can be
completed independently. These components are called fractals. A course is identified by a
unique number.
A regular course has fourteen hours of classroom engagement per credit. A regular course can
be under elective or core compulsory categories. Elective courses of lower levels will run if a
minimum of ten students register for the course. A seven hundred or eight hundred level
elective course can run with a minimum of five registered students.
A self-study course will be from the list of courses approved by the Senate. A student may be
allowed to register for an elective course as a self-study course provided that the course is not
running in that semester as a regular course or a student is not on campus during the course
with prior approval of the competent authority. Not exceeding ten percent of the total graded
credit can be opted as a self-study course.
Independent study enables a student to pursue course credit on an academic topic of interest
under the supervision of a faculty member. Self-discipline and having a sense of own direction
and goals are fundamental requirements of an Independent Study course. A student having
CGPA more than or equal to six can opt for Independent Study courses with the recommendation
of the Supervising Professor.
Each course is conducted by the instructor with the assistance of the required number of tutors
as needed. The instructor is responsible for conducting the course evaluating the performance
of students awarding the grades at the end of the semester and transmitting the grades. The
student shall have access to his or her answer papers of all the written examinations conducted
for a given course. The instructor must keep all evaluation records in his custody at least for the
next six months.
A student should have full attendance in each course. Unless the student takes leave of absence
for valid reasons the student has to attend every lecture tutorial or lab session. The attendance
records must be made available to the student after every lecture. Even if the student attendance
falls below seventy-five percent the student will be allowed to appear for the exams. Students
not meeting the attendance criterion of seventy-five percent will be required to score at least C
grade to pass a course.
The performance of students in a course is evaluated continuously using their interaction in the
classroom and performances in examinations the laboratory work if any and term-papers and
projects. Minor one Minor two and Major examinations are mandatory components of the
evaluation of a regular fourteen-week long lecture course. The minor examinations shall be of
sixty minutes duration and the major examination of one hundred twenty minutes. The total
weightage of the examination component shall be between forty percent to sixty percent of the
total weightage of evaluation measure. Evaluation policy has to be known from the first day of
the class.
At the end of the semester a student is awarded a relative letter grade in each course by the
Instructor offering the course considering the performance of the student during the semester
with respect to those of the other students registered in the course. Ten regular letter grades
namely A star A A minus B B minus C C minus D E and F shall be awarded in each course. Each
letter grade is associated with a numerical equivalent on a ten-point scale. In addition there are
four special letter grades namely I S X and U which stand for Incomplete Satisfactory Thesis
Continuation and Unsatisfactory respectively.
The Semester Grade Point Average SGPA is a weighted average of the grades earned by a
student in all courses credited by her or him and reflects her or his academic performance in the
respective semester. The Cumulative Grade Point Average CGPA indicates the overall academic
performance of a student in all the courses registered up to the latest completed semester. The
CGPA is calculated on the basis of all pass grades obtained in all completed semesters.
A Grade Card shall be issued to each student at the end of each semester and a Transcript at
the end of the Programme. IIT Jodhpur Grade Card and Transcript will only indicate the courses
credits and grades completed at IIT Jodhpur.
A student who has missed minor or the major examination due to genuine reasons like illness
may be permitted to write a make-up examination for the missed components. The student
should make an application to the Dean Academics through the course instructor within ten
days from the date of the missed examination explaining the reasons for their absence.
The Indian Institute of Technology Jodhpur currently offers the following Postgraduate
programmes: Master of Science MSc Dual degree MSc MTech Master of Technology MTech
Dual degree MTech PhD and Doctor of Philosophy PhD. The process of admission for all the
programs offered by the Institute normally is underway during April May for Semester one and
during November December for Semester two. In addition the department may process
applications for admissions to PhD programs on a continuous basis and admit students as per
the existing procedure.
A candidate can be admitted in any one of the following categories: Full-time regular Full-time
sponsored Part-time External Part-time online Part-time Project sponsored and Executive.
The admission to the MSc Programs offered by the Institute will be through the Joint Admission
Test for Master JAM organized by IITs or based on written test and or interview conducted by
IIT Jodhpur. For MTech admission the applicant must have a bachelor degree in engineering or
science or a master degree in science MCA Pharmacy Medical Sciences Agricultural Sciences
and Veterinary Sciences. The applicant must either have a valid GATE score or exempted from
GATE as per MHRD circular.
For PhD admission candidates must have a master degree in engineering pharmacy agricultural
science science humanities social sciences management with at least sixty percent marks or at
least six point zero CGPA. Alternatively the applicant must have a bachelor degree of minimum
four year duration in engineering or science with at least seventy percent marks or at least seven
point zero CGPA.
A MSc student can register for a maximum of twenty-four credits and a minimum of ten credits
in a semester. A MTech full-time regular student can register for a maximum of seventeen
credits and a minimum of ten credits per regular semester. A PhD full-time regular student can
register for a maximum of sixteen credits before and after qualifier with a minimum of six credits.
The maximum duration for MSc is eight registered semesters. For MTech regular and part-time
the maximum is twelve registered semesters. For PhD and MTech-PhD dual degree the maximum
is fourteen registered semesters.
The minimum CGPA requirement for continuation and graduation in MTech is five point zero.
For PhD the minimum CGPA for continuation is six point five and seven point five at completion
of course work. A PhD student would be eligible for enhancement in the scholarship after two
years provided the student has met the qualifier requirement.
The qualifier requirements for a PhD student consist of four components: completed course
requirements with minimum CGPA cleared comprehensive examination successful presentation
on state of the art of the chosen field of research and successful presentation and defense of
Research proposal. PhD students are normally expected to complete qualifier requirements
before the beginning of the fifth semester.
The comprehensive examination must be designed to test the general capability of the student
and the breadth of knowledge in the discipline and areas related to the field of research. The
Comprehensive Examination shall be conducted by a Comprehensive Examination Committee
of the Department. The Comprehensive Examination will consist of a written test with three
papers.
Every PhD student is required to give a general seminar in the Department covering the State
of Art of the area of research. The presentation must also include patent landscaping of the area.
The presentation will be evaluated by the Student Research Committee SRC. A candidate must
have a satisfactory grade in SOTA awarded by SRC for becoming eligible to present the research
proposal.
Academic progress of all registered PhD students will be reviewed once in a year by the SRC
and peers in the institute. The review will be based upon Annual Progress Report and a Poster
Presentation open to all. Two consecutive non-satisfactory grades will lead to termination of
registration.
The supervisor shall propose a list of ten examiners consisting of at least four Faculty Members
from abroad and four Faculty Members from India who have the expertise in the area in which
the student undertook research. The examiners shall have at least six years Post-PhD experience.
The Chairman Senate shall select minimum two examiners from the proposed list of which at
least one should be an examiner from outside India.
A student is deemed to have completed the requirements for graduation if she or he has met
the minimum residential requirement earned the minimum credits with the required CGPA
obtained No Dues Clearance and has no case of disciplinary action pending. A student who
completes all the graduation requirements is recommended by the Senate to the Board of
Governors for the award of the appropriate degree in the ensuing convocation.
The student should have at least two publications in Scopus indexed journals and or A plus
conferences for Engineering and Science Departments before submitting the PhD thesis.
Students should submit the synopsis and thesis within six months of the open seminar date.
The Senate of IIT Jodhpur reserves the right to modify or amend without notice Rules and
Regulations of the Postgraduate programmes at IIT Jodhpur.
"""


def is_english_text(text):
    # >90% ASCII characters is treated as English
    if not text.strip():
        return False
    return (sum(1 for c in text if ord(c) < 128) / len(text)) > 0.90


def clean_raw_text(text):
    # Remove URLs, emails, non-ASCII, excessive whitespace and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{5,}\b', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:\'\-]', ' ', text)
    return text.strip()


def scrape_page(url):
    # Fetch URL and extract visible text; returns empty string on failure
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        texts = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "article", "section"]):
            cleaned = clean_raw_text(tag.get_text(separator=" "))
            if is_english_text(cleaned) and len(cleaned.split()) > 5:
                texts.append(cleaned)
        return " ".join(texts)
    except Exception:
        return ""


def collect_corpus(output_path="raw_corpus.txt"):
    documents = []

    # Always include the academic regulation document (embedded from official PDF)
    reg_text = re.sub(r'\s+', ' ', ACADEMIC_REGULATION_TEXT).strip()
    documents.append(reg_text)

    # Live scrape department, academic program, and research pages
    for category, urls in SOURCES.items():
        for url in urls:
            text = scrape_page(url)
            if text and len(text.split()) >= MIN_WORDS:
                documents.append(text)
            time.sleep(1.0)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.strip() + "\n")

    return documents


if __name__ == "__main__":
    collect_corpus(output_path="raw_corpus.txt")
