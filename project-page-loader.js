const params = new URLSearchParams(location.search);
const projectId = params.get("project-id");

class ProjectData {
	constructor(name, description, images) {
		this.name = name;
		this.description = description;
		this.images = images;
	}
}

function getProjectData() {
	switch (projectId) {
		case "beat-em-up":
			return new ProjectData("BEAT 'Em Up", "projects/beat-em-up/description.html", []);
		case "news":
			return new ProjectData("Faking News", "projects/news/description.html", []);
		case "snow":
			return new ProjectData("Ice To Beat You", "projects/snow/description.html", []);
		case "obscurum":
			return new ProjectData("Obscurum", "projects/obscurum/description.html", []);
		case "o2":
			return new ProjectData("O2", "projects/o2/description.html", []);
		case "webgl":
			return new ProjectData("WebGL Based Engine", "projects/webgl/description.html", []);
		case "physics":
			return new ProjectData("3D Physics", "projects/physics/description.html", []);
		case "raytrace":
			return new ProjectData("Raytracer", "projects/raytrace/description.html", []);
	}
}

window.onload = () => {
	const project = getProjectData();
	if (project) {
		document.getElementById("project-title").innerText = project.name;

		fetch(project.description).then(async response => {
			document.getElementById("project-description").innerHTML = await response.text();
		});
	}
}