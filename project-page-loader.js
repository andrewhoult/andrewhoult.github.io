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
			return new ProjectData("BEAT 'Em Up", "This is a semester long group project", []);
		case "news":
			return new ProjectData("Faking News", "This is a semester long group project", []);
		case "snow":
			return new ProjectData("Ice To Beat You", "This is a semester long group project", []);
		case "obscurum":
			return new ProjectData("Obscurum", "This is a semester long group project", []);
		case "o2":
			return new ProjectData("O2", "This is a semester long group project", []);
		case "webgl":
			return new ProjectData("WebGL Based Engine", "This is a semester long group project", []);
		case "physics":
			return new ProjectData("3D Physics", "This is a semester long group project", []);
		case "raytrace":
			return new ProjectData("Raytracer", "This is a semester long group project", []);
	}
}

window.onload = () => {
	const project = getProjectData();
	if (project) {
		document.getElementById("project-title").innerText = project.name;
		document.getElementById("project-description").innerText = project.description;
	}
}