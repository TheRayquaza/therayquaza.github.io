package mlspot.backend.data.repository;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import mlspot.backend.data.model.BlogModel;

import javax.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class BlogRepository implements PanacheRepository<BlogModel> {
}
