const params = new URLSearchParams(location.search);
const projectId = params.get("project-id");

document.addEventListener("DOMContentLoaded", () => {
	const project = getProjectData();
	if (project) {
		document.getElementById("project-title").innerText = project.name;

		fetch(`/projects/${project.projectId}/description.html`).then(async response => {
			document.getElementById("project-description").innerHTML = await response.text();
		});
	}
});

window.addEventListener("load", () => {
	const card = document.getElementById(projectId);

	card.scrollIntoView({ block: "nearest" });
	card.classList.add("active-project");
});

class ProjectData {
	constructor(name, projectId) {
		this.name = name;
		this.projectId = projectId;
	}
}

function getProjectData() {
	switch (projectId) {
		case "beat-em-up":
			return new ProjectData("BEAT 'Em Up", projectId);
		case "news":
			return new ProjectData("Faking News", projectId);
		case "snow":
			return new ProjectData("Ice To Beat You", projectId);
		case "obscurum":
			return new ProjectData("Obscurum", projectId);
		case "o2":
			return new ProjectData("O2", projectId);
		case "webgl":
			return new ProjectData("WebGL Based Engine", projectId);
		case "physics":
			return new ProjectData("3D Physics", projectId);
		case "raytrace":
			return new ProjectData("Raytracer", projectId);
	}
}