window.onload = populateProjects;

function populateProjects() {
	const cards = [
		new ProjectCard(
			"A simple test project.",
			"testproject",
			"https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg",
		),
		new ProjectCard(
			"A very simple test project.",
			"verytestproject",
			"https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg",
		),
	];

	cards.forEach(card => {
		document.getElementById("projects-list").innerHTML += card.getHTML();
	});
}