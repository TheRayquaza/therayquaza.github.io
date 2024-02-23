package mlspot.backend.presentation.rest;

import mlspot.backend.errors.UnauthorizedError;
import mlspot.backend.presentation.rest.request.AdminRequest;
import mlspot.backend.presentation.rest.response.SuccessResponse;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

@Path("/admin")
@Produces(MediaType.APPLICATION_JSON)
public class AdminEndpoint {
    private final Logger logger = LoggerFactory.getLogger(BlogEndpoint.class);

    @ConfigProperty(name = "api.key")
    String apiKey;

    @POST
    @Path("/")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response validateAdmin(@RequestBody AdminRequest adminRequest) {
        logger.info("[POST] /admin");
        if (!adminRequest.getApiKey().equals(apiKey)) {
            logger.warn("A user tried to access admin panel but failed !");
            return Response.status(401).entity(new UnauthorizedError("I see what you are doing, don't try to fool me", 401)).build();
        }
        return Response.status(200).entity(new SuccessResponse()).build();
    }
}