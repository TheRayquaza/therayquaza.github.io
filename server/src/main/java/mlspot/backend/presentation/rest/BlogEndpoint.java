package mlspot.backend.presentation.rest;

import mlspot.backend.converter.EntityResponseConverter;
import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.domain.service.BlogService;
import mlspot.backend.errors.BlogContentNotFoundError;
import mlspot.backend.errors.BlogNotFoundError;
import mlspot.backend.presentation.rest.request.CreateBlogContentRequest;
import mlspot.backend.presentation.rest.request.CreateBlogRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogContentRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogRequest;
import mlspot.backend.presentation.rest.response.BlogContentResponse;
import mlspot.backend.presentation.rest.response.BlogResponse;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;

import static mlspot.backend.converter.EntityResponseConverter.Of;

@Path("/blogs")
@Produces(MediaType.APPLICATION_JSON)
public class BlogEndpoint {
    private final Logger logger = LoggerFactory.getLogger(BlogEndpoint.class);

    @Inject
    BlogService blogService;

    @GET
    @Path("/")
    public Response getAllGBlogsEndpoint() {
        logger.info("[GET] /blogs");
        List<BlogEntity> blogEntities = blogService.getAllBlogs();
        List<BlogResponse> blogResponses = new ArrayList<>();
        blogEntities.forEach(b -> blogResponses.add(Of(b)));
        return Response
                .status(200)
                .entity(blogResponses)
                .build();
    }

    @POST
    @Path("/")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response createBlogEndpoint(@RequestBody CreateBlogRequest createBlogRequest) {
        logger.info("[POST] /blogs");
        if (createBlogRequest == null || createBlogRequest.getTitle() == null)
            return Response.status(400).build();
        BlogEntity blogEntity = blogService.createBlog(createBlogRequest.getTitle());
        return Response.status(200).entity(Of(blogEntity)).build();
    }


    @GET
    @Path("/{blogId}")
    public Response getBlogEndpoint(@PathParam(value = "blogId") Long blogId) {
        logger.info("[GET] /blogs/" + blogId);
        if (blogId == null)
            return Response.status(400).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.getBlog(blogId))).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] could not find blog " + blogId);
            return Response.status(404).build();
        }
    }

    @DELETE
    @Path("/{blogId}")
    public Response deleteBlogEndpoint(@PathParam(value = "blogId") Long blogId) {
        logger.info("[DELETE] /blogs/" + blogId);
        if (blogId == null)
            return Response.status(400).build();
        try {
            if (blogService.deleteBlog(blogId))
                return Response.status(200).build();
            return Response.status(400).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] could not find blog " + blogId);
            return Response.status(404).build();
        }
    }

    @PUT
    @Path("/{blogId}")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response modifyBlogEndpoint(@RequestBody ModifyBlogRequest request, @PathParam(value = "blogId") Long blogId) {
        logger.info("[PUT] /blogs/" + blogId);
        if (blogId == null || request == null)
            return Response.status(400).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.modifyBlog(request, blogId))).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] could not find blog " + blogId);
            return Response.status(404).build();
        }
    }

    @GET
    @Path("/{blogId}/content")
    public Response getAllBlogContentEndpoint(@PathParam(value = "blogId") Long blogId) {
        logger.info("[GET] /blogs/" + blogId + "/content");
        if (blogId == null)
            return Response.status(400).build();
        try {
            List<BlogContentEntity> list = blogService.getAllBlogContent(blogId);
            List<BlogContentResponse> response = new ArrayList<>();
            list.forEach(b -> response.add(EntityResponseConverter.Of(b)));
            return Response.status(200).entity(response).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).build();
        }
    }

    @POST
    @Path("/{blogId}/content")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response createBlogContentEndpoint(@RequestBody CreateBlogContentRequest request, @PathParam(value = "blogId") Long blogId) {
        logger.info("[POST] /blogs/" + blogId + "/content");
        if (blogId == null || request == null)
            return Response.status(400).build();
        try {
            return Response.status(200).entity(Of(blogService.createBlogContent(request, blogId))).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] blog with id " + blogId + " could not be found");
            return Response.status(404).build();
        }
    }

    @GET
    @Path("/{blogId}/content/{contentId}")
    public Response getBlogContentEndpoint(@PathParam(value = "blogId") Long blogId, @PathParam(value = "contentId") Long contentId) {
        logger.info("[GET] /blogs/" + blogId + "/content/" + contentId);
        if (blogId == null || contentId == null)
            return Response.status(400).build();
        try {
            return Response.status(200).entity(Of(blogService.getBlogContent(blogId, contentId))).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).build();
        } catch (BlogContentNotFoundError ignored) {
            logger.info("[404] blog content with id " + contentId + " could not be found");
            return Response.status(404).build();
        }
    }

    @DELETE
    @Path("/{blogId}/content/{contentId}")
    public Response deleteBlogContentEndpoint(@PathParam(value = "blogId") Long blogId, @PathParam(value = "contentId") Long contentId) {
        logger.info("[DELETE] /blogs/" + blogId + "/content/" + contentId);
        if (blogId == null || contentId == null)
            return Response.status(400).build();
        try {
            if (blogService.deleteBlogContent(blogId, contentId))
                return Response.status(200).build();
            return Response.status(400).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).build();
        } catch (BlogContentNotFoundError ignored) {
            logger.info("[404] blog content with id " + contentId + " could not be found");
            return Response.status(404).build();
        }
    }

    @PUT
    @Path("/{blogId}/content/{contentId}")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response modifyBlogContentEndpoint(@RequestBody ModifyBlogContentRequest request, @PathParam(value = "blogId") Long blogId, @PathParam(value = "contentId") Long contentId) {
        logger.info("[PUT] /blogs/" + blogId + "/content/" + contentId);
        if (blogId == null || contentId == null || request == null)
            return Response.status(400).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.modifyBlogContent(request, blogId, contentId))).build();
        } catch (BlogNotFoundError ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).build();
        } catch (BlogContentNotFoundError ignored) {
            logger.info("[404] blog content with id " + contentId + " not found");
            return Response.status(404).build();
        }
    }

}