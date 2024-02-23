package mlspot.backend.domain.service;

import mlspot.backend.converter.ModelEntityConverter;
import mlspot.backend.data.model.ProjectModel;
import mlspot.backend.data.repository.ProjectRepository;
import mlspot.backend.domain.entity.ProjectEntity;
import mlspot.backend.exceptions.ProjectNotFoundException;
import mlspot.backend.presentation.rest.request.ModifyProjectRequest;
import mlspot.backend.utils.DateUtil;

import javax.enterprise.context.ApplicationScoped;
import javax.inject.Inject;
import javax.transaction.Transactional;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@ApplicationScoped
public class ProjectService {
    @Inject
    ProjectRepository projectRepository;

    @Transactional
    public List<ProjectEntity> getAllProjects() {
        List<ProjectEntity> list = new ArrayList<>();
        projectRepository.findAll().stream().forEach(p -> {
            list.add(ModelEntityConverter.Of(p));
        });
        return list;
    }

    @Transactional
    public ProjectEntity getProject(Long projectId) throws ProjectNotFoundException {
        ProjectModel projectModel = projectRepository.findById(projectId);
        if (projectModel == null)
            throw new ProjectNotFoundException();
        return ModelEntityConverter.Of(projectModel);
    }

    @Transactional
    public ProjectEntity createProject(String name) {
        ProjectModel projectModel = new ProjectModel().withName(name);
        projectModel.persist();
        return ModelEntityConverter.Of(projectModel);
    }

    @Transactional
    public boolean deleteProject(Long projectId) throws ProjectNotFoundException {
        ProjectModel projectModel = projectRepository.findById(projectId);
        if (projectModel == null)
            throw new ProjectNotFoundException();
        return projectRepository.deleteById(projectId);
    }

    @Transactional
    public ProjectEntity modifyProject(ModifyProjectRequest request, Long projectId) throws ProjectNotFoundException {
        ProjectModel projectModel = projectRepository.findById(projectId);
        if (projectModel == null)
            throw new ProjectNotFoundException();
        if (request.getFinishedDate() != null && DateUtil.isValidLocalDate(request.getFinishedDate()))
            projectModel.setFinishedDate(LocalDate.parse(request.getFinishedDate()));
        if (request.getStartingDate() != null && DateUtil.isValidLocalDate(request.getStartingDate()))
            projectModel.setStartingDate(LocalDate.parse(request.getStartingDate()));
        if (request.getTechnologies() != null)
            projectModel.setTechnologies(String.join(",", request.getTechnologies()));
        if (request.getLink() != null)
            projectModel.setLink(request.getLink());
        if (request.getMembers() != null)
            projectModel.setMembers(request.getMembers());
        if (request.getName() != null)
            projectModel.setName(request.getName());
        if (request.getDescription() != null)
            projectModel.setDescription(request.getDescription());

        return ModelEntityConverter.Of(projectModel);
    }
}
