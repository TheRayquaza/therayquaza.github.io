package mlspot.backend.data.repository;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import mlspot.backend.data.model.ProjectModel;

import javax.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class ProjectRepository implements PanacheRepository<ProjectModel> {
}
