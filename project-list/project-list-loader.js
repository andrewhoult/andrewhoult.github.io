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
		<fluid-box id="fluid-box-project-${this.id}">
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
		</fluid-box>
		`
	}
}

function populateProjects() {
	const longCards = [
		new ProjectCard(
			"Shadows Beneath the Dust",
			"A Lethal Company inspired 4 player co-op game. Loot the last remaining scraps from long abandoned mines.",
			"spaghetti",
			"/projects/spaghetti/images/thumbnail.png",
		),
		new ProjectCard(
			"BEAT 'Em Up",
			"A rythm-based beat 'em up. Fight rival musically-themed gangs to reclaim you turf in Funky Town.",
			"beat-em-up",
			"/projects/beat-em-up/images/thumbnail.png",
		),
	];

	const jamCards = [
		new ProjectCard(
			"Ice To Beat You",
			"This game is a unique mix of physical and digital. Battle it out in a legendary snowball fight against the neighborhood hooligans. With real fake snowballs!",
			"snow",
			"/projects/snow/images/thumbnail.png",
		),
		new ProjectCard(
			"Obscurum",
			"A sound-focused horror game where you can only feel your way around using your trusty poking stick.",
			"obscurum",
			"/projects/obscurum/images/thumbnail.png",
		),
		new ProjectCard(
			"Faking News",
			"Make up headlines, find photos for the cover, and broadcast your stories to your friends. The result of a 48 hour game jam with a group of 5.",
			"faking-news",
			"/projects/faking-news/images/thumbnail.png",
		),
		new ProjectCard(
			"O2",
			"Delve into the depth of a mysterious planet, but make sure to manage your limited oxygen. Made during a 48 hour game jam with a group of 3.",
			"o2",
			"/projects/o2/images/thumbnail.png",
		),
	];

	const soloCards = [
		new ProjectCard(
			"2D Platformer",
			"A really simple 2D platformer written in Vulkan, written completely from scratch with only the Vulkan and Windows SDKs.",
			"platformer",
			"/projects/platformer/images/jump1.png",
		),
		new ProjectCard(
			"WebGL Engine",
			"A 3D engine written in TypeScript with multiplayer support, node.js server hosting, and a web-based level editor.",
			"webgl",
			"/projects/webgl/images/sci1.png",
		),
		new ProjectCard(
			"Raytraced Minecraft",
			"A simple realtime raytracing engine using textures from Minecraft.",
			"raytrace",
			"/projects/raytrace/images/thumbnail.png",
		),
		new ProjectCard(
			"3D Physics",
			"A simple 3D physics engine.",
			"physics",
			"/projects/physics/images/thumbnail1.png",
		),
	];

	addCards(longCards, "projects-list-long");
	addCards(jamCards, "projects-list-jam");
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