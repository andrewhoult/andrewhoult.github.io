const params = new URLSearchParams(location.search);
const projectId = params.get("project-id");

document.addEventListener("DOMContentLoaded", () => {
	const project = getProjectData();
	if (project) {
		document.getElementById("project-title").innerText = project.name;

		fetch(project.root + "/description.html").then(async response => {
			document.getElementById("project-description").innerHTML = await response.text();
		});

		project.imagesURL.forEach(url => {
			addImage(project.root + "/" + url);
		});
	}
});

window.addEventListener("load", () => {
	const card = document.getElementById(projectId);

	card.scrollIntoView({ block: "nearest" });
	card.style.background = "white";
});

class ProjectData {
	constructor(name, root, imagesURL) {
		this.name = name;
		this.root = root;
		this.imagesURL = imagesURL;
	}
}

function getProjectData() {
	switch (projectId) {
		case "beat-em-up":
			return new ProjectData("BEAT 'Em Up", `/projects/${projectId}`, ["images/thumbnail.png", "images/thumbnail.png", "images/thumbnail.png"]);
		case "news":
			return new ProjectData("Faking News", `/projects/${projectId}`, []);
		case "snow":
			return new ProjectData("Ice To Beat You", `/projects/${projectId}`, []);
		case "obscurum":
			return new ProjectData("Obscurum", `/projects/${projectId}`, []);
		case "o2":
			return new ProjectData("O2", `/projects/${projectId}`, []);
		case "webgl":
			return new ProjectData("WebGL Based Engine", `/projects/${projectId}`, []);
		case "physics":
			return new ProjectData("3D Physics", `/projects/${projectId}`, []);
		case "raytrace":
			return new ProjectData("Raytracer", `/projects/${projectId}`, []);
	}
}