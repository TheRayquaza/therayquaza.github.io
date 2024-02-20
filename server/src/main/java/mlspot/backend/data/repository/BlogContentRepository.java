package mlspot.backend.data.repository;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import mlspot.backend.data.model.BlogContentModel;

import javax.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class BlogContentRepository implements PanacheRepository<BlogContentModel> {
}
