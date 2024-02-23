package mlspot.backend.data.repository;

import javax.enterprise.context.ApplicationScoped;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import mlspot.backend.data.model.BlogCategoryModel;

@ApplicationScoped
public class BlogCategoryRepository implements PanacheRepository<BlogCategoryModel> {   
}
