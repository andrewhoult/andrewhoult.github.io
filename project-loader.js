window.onload = populateProjects;

function populateProjects() {
	const collabCards = [
		new ProjectCard(
			"BEAT 'Em Up",
			"A rythm-based beat 'em up. Fight rival musically themed gangs to reclaim you turf in Funky Town. Made as a semester long project with a group of 5.",
			"projects/beat-em-up/page.html",
			"projects/beat-em-up/images/thumbnail.png",
		),
		new ProjectCard(
			"Ice To Beat You",
			"A mix of physical and digital. Battle it out in a snowball fight against the neighborhood hooligans using real fake snowballs! Made during a 4 day game jame with a group of 6.",
			"projects/snow/page.html",
			"projects/snow/images/thumbnail.png",
		),
		new ProjectCard(
			"Obscurum",
			"A sound intensive horror game where you can only see by using your trusty poking stick. Made during a 4 day game jam with a group of 5.",
			"projects/sound/page.html",
			"projects/sound/images/thumbnail.png",
		),
		new ProjectCard(
			"Faking News",
			"Make up headlines, find photos for the cover, and broadcast your stories to your friends. The result of a 48 hour game jam with a group of 5.",
			"projects/faking-news/page.html",
			"projects/faking-news/images/thumbnail.png",
		),
		new ProjectCard(
			"O2",
			"Delve into the depth of a mysterious planet, but make sure to manage your limited oxygen. Made during a 48 hour game jam with a group of 3.",
			"projects/o2/page.html",
			"projects/o2/images/thumbnail.png",
		),
	];

	const soloCards = [
		new ProjectCard(
			"WebGL Based Engine",
			"A 3D engine written in TypeScript with multiplayer support, node.js server hosting, and a web-based level editor.",
			"projects/web3d/page.html",
			"projects/web3d/images/thumbnail.png",
		),
		new ProjectCard(
			"Raytracer",
			"A simple realtime raytracing engine.",
			"projects/raytrace/page.html",
			"projects/raytrace/images/thumbnail.png",
		),
		new ProjectCard(
			"3D Physics",
			"A simple 3D physics engine.",
			"projects/physics/page.html",
			"projects/physics/images/thumbnail.png",
		),
	];

	addCards(collabCards, "projects-list-collab");
	addCards(soloCards, "projects-list-solo");
}

function addCards(cards, id) {
	cards.forEach(card => {
		document.getElementById(id).innerHTML += card.getHTML();
	});
}