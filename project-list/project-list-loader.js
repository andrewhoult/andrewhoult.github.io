document.addEventListener("DOMContentLoaded", populateProjects);

class ProjectCard {
	constructor(title, description, id, imageUrl) {
		this.title = title;
		this.description = description;
		this.id = id;
		this.imageUrl = imageUrl;
	}

	getHTML() {
		return `
		<a href="/project-page/project-page.html?project-id=${this.id}">
			<div class="project-card-container">
				<div class="project-card" id="${this.id}">
					<img src="${this.imageUrl}" alt="Project thumbnail">
					<div>
						<strong>${this.title}</strong>
						<p>${this.description}</p>
					</div>
				</div>
			</div>
		</a>
		`
	}
}

function populateProjects() {
	const collabCards = [
		new ProjectCard(
			"BEAT 'Em Up",
			"A rythm-based beat 'em up. Fight rival musically-themed gangs to reclaim you turf in Funky Town.",
			"beat-em-up",
			"https://media.githubusercontent.com/media/andrewhoult/andrewhoult.github.io/main/projects/beat-em-up/images/thumbnail.png",
		),
		new ProjectCard(
			"Ice To Beat You",
			"This game is a unique mix of physical and digital. Battle it out in a legendary snowball fight against the neighborhood hooligans. With real fake snowballs!",
			"snow",
			"https://media.githubusercontent.com/media/andrewhoult/andrewhoult.github.io/main/projects/snow/images/thumbnail.png",
		),
		new ProjectCard(
			"Obscurum",
			"A sound-focused horror game where you can only feel your way around using your trusty poking stick.",
			"obscurum",
			"https://media.githubusercontent.com/media/andrewhoult/andrewhoult.github.io/main/projects/obscurum/images/thumbnail.png",
		),
		// new ProjectCard(
		// 	"Faking News",
		// 	"Make up headlines, find photos for the cover, and broadcast your stories to your friends. The result of a 48 hour game jam with a group of 5.",
		// 	"news",
		// 	"https://media.githubusercontent.com/media/andrewhoult/andrewhoult.github.io/main/projects/faking-news/images/thumbnail.png",
		// ),
		// new ProjectCard(
		// 	"O2",
		// 	"Delve into the depth of a mysterious planet, but make sure to manage your limited oxygen. Made during a 48 hour game jam with a group of 3.",
		// 	"o2",
		// 	"https://media.githubusercontent.com/media/andrewhoult/andrewhoult.github.io/main/projects/o2/images/thumbnail.png",
		// ),
	];

	const soloCards = [
		new ProjectCard(
			"WebGL Based Engine",
			"A 3D engine written in TypeScript with multiplayer support, node.js server hosting, and a web-based level editor.",
			"webgl",
			"projects/web3d/images/thumbnail.png",
		),
		new ProjectCard(
			"Raytracer",
			"A simple realtime raytracing engine.",
			"raytrace",
			"projects/raytrace/images/thumbnail.png",
		),
		new ProjectCard(
			"3D Physics",
			"A simple 3D physics engine.",
			"physics",
			"https://media.githubusercontent.com/media/andrewhoult/andrewhoult.github.io/main/projects/physics/images/thumbnail.png",
		),
	];

	addCards(collabCards, "projects-list-collab");
	addCards(soloCards, "projects-list-solo");
}

function addCards(cards, id) {
	let parent = document.getElementById(id);
	if (parent) {
		cards.forEach(card => {
			parent.innerHTML += card.getHTML();
		});
	}
}