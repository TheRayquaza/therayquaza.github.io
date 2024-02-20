package mlspot.backend.domain.service;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import mlspot.backend.converter.ModelEntityConverter;
import mlspot.backend.data.model.BlogContentModel;
import mlspot.backend.data.model.BlogModel;
import mlspot.backend.data.repository.BlogContentRepository;
import mlspot.backend.data.repository.BlogRepository;
import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.errors.BlogContentNotFoundError;
import mlspot.backend.errors.BlogNotFoundError;
import mlspot.backend.presentation.rest.request.CreateBlogContentRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogContentRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogRequest;
import org.aesh.readline.editing.EditMode;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;

import javax.enterprise.context.ApplicationScoped;
import javax.inject.Inject;
import javax.transaction.Transactional;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@ApplicationScoped
public class BlogService {
    @Inject
    BlogRepository blogRepository;

    @Inject
    BlogContentRepository blogContentRepository;

    @Transactional
    public List<BlogEntity> getAllBlogs() {
        List<BlogModel> blogModels = blogRepository.findAll().stream().toList();
        List<BlogEntity> blogEntities = new ArrayList<>();
        blogModels.forEach(b -> blogEntities.add(ModelEntityConverter.Of(b)));
        return blogEntities;
    }

    @Transactional
    public BlogEntity getBlog(Long blogId) throws BlogNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        return ModelEntityConverter.Of(blogModel);
    }

    @Transactional
    public BlogEntity createBlog(String title) {
        BlogModel blogModel = new BlogModel().withTitle(title);
        blogModel.persist();
        return ModelEntityConverter.Of(blogModel);
    }

    @Transactional
    public boolean deleteBlog(Long blogId) throws BlogNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        if (!blogRepository.deleteById(blogId))
            return false;
        blogContentRepository
                .findAll()
                .stream()
                .filter(b -> Objects.equals(b.getBlogId(), blogId))
                .forEach(PanacheEntityBase::delete);
        return true;
    }

    @Transactional
    public BlogEntity modifyBlog(ModifyBlogRequest request, Long blogId) throws BlogNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        if (request.getTitle() != null)
            blogModel.setTitle(request.getTitle());
        if (request.getDescription() != null)
            blogModel.setDescription(request.getDescription());
        return ModelEntityConverter.Of(blogModel);
    }

    @Transactional
    public List<BlogContentEntity> getAllBlogContent(Long blogId) throws BlogNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        List<BlogContentModel> blogContentModels = blogContentRepository
                .findAll()
                .stream()
                .filter(b -> b.getBlogId().equals(blogId))
                .toList();
        List<BlogContentEntity> contentEntities = new ArrayList<>();
        blogContentModels.forEach(b -> contentEntities.add(ModelEntityConverter.Of(b)));
        return contentEntities;
    }

    @Transactional
    public BlogContentEntity getBlogContent(Long blogId, Long contentId) throws BlogNotFoundError, BlogContentNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        BlogContentModel blogContentModel = blogContentRepository.findById(contentId);
        if (blogContentModel == null)
            throw new BlogContentNotFoundError();
        if (!Objects.equals(blogContentModel.getBlogId(), blogId))
            throw new BlogContentNotFoundError();
        return ModelEntityConverter.Of(blogContentModel);
    }

    @Transactional
    public boolean deleteBlogContent(Long blogId, Long contentId) throws BlogNotFoundError, BlogContentNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        BlogContentModel blogContentModel = blogContentRepository.findById(contentId);
        if (blogContentModel == null)
            throw new BlogContentNotFoundError();
        if (!Objects.equals(blogContentModel.getBlogId(), blogId))
            throw new BlogContentNotFoundError();
        blogContentModel.delete();
        return true;
    }

    @Transactional
    public BlogContentEntity createBlogContent(CreateBlogContentRequest request, Long blogId) throws BlogNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        BlogContentModel blogContentModel = new BlogContentModel()
                .withBlogId(blogId)
                .withContent(request.getContent())
                .withType(request.getType());
        blogContentModel.persist();
        return ModelEntityConverter.Of(blogContentModel);
    }

    @Transactional
    public BlogContentEntity modifyBlogContent(ModifyBlogContentRequest request, Long blogId, Long contentId) throws BlogNotFoundError, BlogContentNotFoundError {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundError();
        BlogContentModel blogContentModel = blogContentRepository.findById(contentId);
        if (blogContentModel == null)
            throw new BlogContentNotFoundError();
        if (!Objects.equals(blogContentModel.getBlogId(), blogId))
            throw new BlogContentNotFoundError();
        if (request.getType() != null)
            blogContentModel.setType(request.getType());
        if (request.getContent() != null)
            blogContentModel.setContent(request.getContent());
        return ModelEntityConverter.Of(blogContentModel);
    }
}
